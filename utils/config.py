from argparse import Action, ArgumentParser, Namespace
import copy
from easydict import EasyDict
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


def parse_cfg(yaml_file, update_dict={}):
    """load configurations from a yaml file & update from command-line argments
    Input:
        yaml_file: str, path to a yaml configuration file
        update_dict: dict, to modify/update options in those yaml configurations
    Output:
        cfg: EasyDict
    """
    with open(args.cfg, "r") as f:
        cfg = EasyDict(yaml.safe_load(f))

    if update_dict:
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
        if isinstance(v, EasyDict):
            d[k] = easydict2dict(v)
        else:
            d[k] = v
    return d


if "__main__" == __name__:
    # test command:
    #   python config.py --cfg-options int=5 dict2.lr=8 dict2.newdict.newitem=fly

    import pprint
    parser = ArgumentParser()
    parser.add_argument("--cfg", type=str, default="../config.yaml")
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
    cfg = parse_cfg(args.cfg, args.cfg_options)
    pprint.pprint(cfg)
    with open("backup-config.yaml", 'w') as f:
        yaml.dump(easydict2dict(cfg), f)