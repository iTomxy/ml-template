from argparse import Action, ArgumentParser, Namespace
import copy, os, json
# from easydict import EasyDict
from typing import Any, Optional, Sequence, Tuple, Union
import yaml


"""DictAction from open-mmlab/mmengine
Ref:
    https://github.com/open-mmlab/mmdetection/blob/main/tools/train.py
    https://github.com/open-mmlab/mmengine/blob/main/mmengine/config/config.py
"""


class DictAction(Action):
    """
    argparse action to split an argument into KEY=VALUE form
    on the first = and append to a dictionary. List options can
    be passed as comma separated values, i.e 'KEY=V1,V2,V3', or with explicit
    brackets, i.e. 'KEY=[V1,V2,V3]'. It also support nested brackets to build
    list/tuple values. e.g. 'KEY=[(V1,V2),(V3,V4)]'
    """

    @staticmethod
    def _parse_int_float_bool(val: str) -> Union[int, float, bool, Any]:
        """parse int/float/bool value in the string."""
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        if val.lower() in ['true', 'false']:
            return True if val.lower() == 'true' else False
        if val == 'None':
            return None
        return val

    @staticmethod
    def _parse_iterable(val: str) -> Union[list, tuple, Any]:
        """Parse iterable values in the string.

        All elements inside '()' or '[]' are treated as iterable values.

        Args:
            val (str): Value string.

        Returns:
            list | tuple | Any: The expanded list or tuple from the string,
            or single value if no iterable values are found.

        Examples:
            >>> DictAction._parse_iterable('1,2,3')
            [1, 2, 3]
            >>> DictAction._parse_iterable('[a, b, c]')
            ['a', 'b', 'c']
            >>> DictAction._parse_iterable('[(1, 2, 3), [a, b], c]')
            [(1, 2, 3), ['a', 'b'], 'c']
        """

        def find_next_comma(string):
            """Find the position of next comma in the string.

            If no ',' is found in the string, return the string length. All
            chars inside '()' and '[]' are treated as one element and thus ','
            inside these brackets are ignored.
            """
            assert (string.count('(') == string.count(')')) and (
                    string.count('[') == string.count(']')), \
                f'Imbalanced brackets exist in {string}'
            end = len(string)
            for idx, char in enumerate(string):
                pre = string[:idx]
                # The string before this ',' is balanced
                if ((char == ',') and (pre.count('(') == pre.count(')'))
                        and (pre.count('[') == pre.count(']'))):
                    end = idx
                    break
            return end

        # Strip ' and " characters and replace whitespace.
        val = val.strip('\'\"').replace(' ', '')
        is_tuple = False
        if val.startswith('(') and val.endswith(')'):
            is_tuple = True
            val = val[1:-1]
        elif val.startswith('[') and val.endswith(']'):
            val = val[1:-1]
        elif ',' not in val:
            # val is a single value
            return DictAction._parse_int_float_bool(val)

        values = []
        while len(val) > 0:
            comma_idx = find_next_comma(val)
            element = DictAction._parse_iterable(val[:comma_idx])
            values.append(element)
            val = val[comma_idx + 1:]

        if is_tuple:
            return tuple(values)

        return values

    def __call__(self,
                 parser: ArgumentParser,
                 namespace: Namespace,
                 values: Union[str, Sequence[Any], None],
                 option_string: str = None):
        """Parse Variables in string and add them into argparser.

        Args:
            parser (ArgumentParser): Argument parser.
            namespace (Namespace): Argument namespace.
            values (Union[str, Sequence[Any], None]): Argument string.
            option_string (list[str], optional): Option string.
                Defaults to None.
        """
        # Copied behavior from `argparse._ExtendAction`.
        options = copy.copy(getattr(namespace, self.dest, None) or {})
        if values is not None:
            for kv in values:
                key, val = kv.split('=', maxsplit=1)
                options[key] = self._parse_iterable(val)
        setattr(namespace, self.dest, options)


INCLUDE_KEY = "__include__"


def load_cfg_file(cfg_file):
    """read from yaml/json & return parsed dict"""
    assert os.path.isfile(cfg_file), cfg_file
    ext = os.path.splitext(cfg_file)[1].lower()
    with open(cfg_file, "r") as f:
        if ".json" == ext:
            cfg = json.load(f)
        else: # .yaml / .yml
            cfg = yaml.safe_load(f)

    return cfg


def inherit(cfg, path_prefix='.'):
    """recursively include(inherit) configurations from other files
    Included files are regarded as `ancestor` who carries common configurations.
    `Descendant` overwrites `ancestor` if duplicated configurations exist.
    Input:
        cfg: dict
        path_prefix: str, the starting path to locate other included configuration files
            in case they are specified by relative paths.
    Output:
        base_cfg: dict, updated
    """
    if INCLUDE_KEY not in cfg:
        return cfg

    includes = cfg[INCLUDE_KEY]
    if isinstance(includes, str):
        includes = [includes]
    assert isinstance(includes, (list, tuple))

    base_cfg = {} # summary of ancestor configurations
    for inc in includes:
        assert isinstance(inc, str)
        if inc.strip() == '':
            continue
        if not os.path.isabs(inc):
            inc = os.path.abspath(os.path.join(path_prefix, inc))

        _cfg = inherit(load_cfg_file(inc), os.path.dirname(inc))
        # NOTE: Duplicated keys should NOT exist in `sibbling` ancestor yaml files,
        # otherwise the overwritting behaviour is undefined.
        base_cfg.update(_cfg) # `update` deals with embedded dict

    # cfg.update(base_cfg) # ancestor overwrites descendant
    # return cfg
    base_cfg.update(cfg) # descendant overwrites ancestor
    return base_cfg


def parse_cfg(cfg_file, *update_dicts):
    """load configurations from file & update from command-line argments
    Input:
        cfg_file: str, path to a configuration file (e.g. yaml, json)
        update_dicts: dict, to modify/update options written in `cfg_file`
    Output:
        cfg: dict
    """
    cfg = load_cfg_file(cfg_file)
    # include configurations in other yaml files indicated by `INCLUDE_KEY`
    cfg = inherit(cfg, os.path.dirname(cfg_file) or '.')

    for update_dict in update_dicts:
        if update_dict is None:
            continue
        assert isinstance(update_dict, dict)
        for k, v in update_dict.items():
            k_list = k.split('.')
            assert len(k_list) > 0
            if len(k_list) == 1:
                cfg[k_list[0]] = v
            else:
                ptr = cfg
                for i, _k in enumerate(k_list):
                    if i == len(k_list) - 1: # last layer
                        ptr[_k] = v
                    elif _k not in ptr:
                        ptr[_k] = {}

                    ptr = ptr[_k]

    return cfg


def easydict2dict(ed):
    """convert EasyDict to dict for clean yaml"""
    d = {}
    for k, v in ed.items():
        if isinstance(v, dict): # EasyDict is also dict
            d[k] = easydict2dict(v)
        elif isinstance(v, (tuple, list)):
            d[k] = [easydict2dict(_v) if isinstance(_v, dict) else _v for _v in v]
        else:
            d[k] = v
    return d


if "__main__" == __name__:
    # test command:
    #   python config.py --cfg-options int=5 dict2.lr=8 dict2.newdict.newitem=fly

    import pprint
    parser = ArgumentParser()
    parser.add_argument("--cfg", type=str, default="../config.yaml")
    parser.add_argument("--flag", action="store_true")
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()

    pprint.pprint(args.cfg_options)
    pprint.pprint(args.__dict__)
    flags = {k: v for k, v in args._get_kwargs() if k not in ("cfg", "cfg_options")}
    pprint.pprint(flags)

    cfg = parse_cfg(args.cfg, flags, args.cfg_options)
    pprint.pprint(cfg)
    # cfg = EasyDict(cfg) # wrap by EasyDict for easy attributes access

    with open("backup-config.yaml", 'w') as f:
        # yaml.dump(cfg, f) # OK
        yaml.dump(easydict2dict(cfg), f) # cleaner yaml

