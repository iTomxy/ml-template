import argparse
from email.header import Header
from email.mime.text import MIMEText
from email.utils import parseaddr, formataddr
import smtplib

"""send E-mail with python

Warning:
Do NOT name this file `email.py` cuz it conflicts
with the package name `email` and that will raise error.
"""

def send_email(
    subject, text, to_name_list, to_addr_list,
    server, from_name, from_addr, password, port=0, ssl=False, debuglevel=1
):
    """send text E-mail
    subject: str, E-mail subject
    text: str, E-mail body
    to_name_list: List[str], list of receivers' name
    to_addr_list: List[str], list of receivers' E-mail address
    server: str, SMTP server IP
    from_name: str, sender's name
    from_addr: str, sender's E-mail address
    password: str, password or authorisation code of the sender account on the server
    port: int = 0, connect to which port of the SMTP server
    ssl: bool = False, use SMTP_SSL instead of SMTP for those servers that require so
    debuglevel: int = 1, set 0 to depress verbose output
    """
    assert isinstance(to_addr_list, (list, tuple)) and len(to_addr_list) > 0
    for i, t in enumerate(to_addr_list):
        assert isinstance(t, str) and '@' in t, "Invalid E-mail address: [{}] ``{}''".format(i + 1, t)
    # assert len(to_name_list) == len(to_addr_list)

    msg = MIMEText(text, 'plain', 'utf-8')
    msg['From'] = formataddr((Header(from_name, 'utf-8').encode(), from_addr)) if from_name else from_addr
    to_list = [formataddr((Header(tn, 'utf-8').encode(), ta)) for tn, ta in zip(to_name_list, to_addr_list)]
    to_list.extend(to_addr_list[len(to_name_list):])
    msg['To'] = ",".join(to_list)

    msg['Subject'] = Header(subject, 'utf-8').encode()

    # with smtplib.SMTP(server, port) as server:
    if ssl:
        server = smtplib.SMTP_SSL(server, port)
    else:
        server = smtplib.SMTP(server, port)
    server.set_debuglevel(debuglevel)
    server.login(from_addr, password)
    server.sendmail(from_addr, to_addr_list, msg.as_string())
    server.quit()


if "__main__" == __name__:
    parser = argparse.ArgumentParser()#fromfile_prefix_chars='@')
    parser.add_argument('text', type=str, nargs='+', metavar="STR", help='email body/content')
    parser.add_argument('-s', '--subject', type=str, nargs='+', metavar="STR", default=["NO SUBJECT"], help='email subject')
    parser.add_argument('-S', '--smtp-server', type=str, metavar="IP", default="", help="SMTP server IP address")
    parser.add_argument('-t', '--to-addr', type=str, nargs='+', metavar="ADDR", default=[], help="receivers' address/account")
    parser.add_argument('-T', '--to-name', type=str, metavar="NAME", nargs='+', default=["Receiver_1"],
        help="receivers' name, separated by space. Concatenate with `_` for multi-word names, e.g. `Jerry_Mouse Spike_Dog`.")
    parser.add_argument('-f', '--from-addr', type=str, metavar="ADDR", default="", help="sender address/account")
    parser.add_argument('-F', '--from-name', type=str, metavar="NAME", default="Sender",
        help="sender name. Concatenate with `_` if multi-word, e.g. `Thomas_Cat`.")
    parser.add_argument('-p', '--password', type=str, metavar="PSW", default="", help="password (or authorisation code) of sender email account")
    parser.add_argument('-P', '--smtp-port', type=int, metavar="PORT", default=0)
    parser.add_argument('--ssl', action="store_true", help="set if using SSL is required by the SMTP server")
    parser.add_argument('-d', '--debug-level', type=int, metavar="INT", default=1)
    args = parser.parse_args()

    assert len(args.to_addr) > 0, "Please specify receiver email address/es, e.g. jerry.mouse@get-cheese.edu"
    assert args.from_addr, "Please specify sender email address, e.g. thomas.cat@catch-mouse.com"
    assert args.password, "Please specify sender account password (or authorisation code)"
    assert args.smtp_server, "Please specify SMTP server, e.g. smtp.sina.cn"

    # deal with name, e.g. `Spike_Dog` -> `Spike Dog`
    args.from_name = args.from_name.replace('_', ' ')
    args.to_name = [tn.replace('_', ' ') for tn in args.to_name]

    send_email(
        ' '.join(args.subject), ' '.join(args.text), args.to_name, args.to_addr,
        args.smtp_server, args.from_name, args.from_addr, args.password, args.smtp_port,
        args.ssl, args.debug_level
    )
