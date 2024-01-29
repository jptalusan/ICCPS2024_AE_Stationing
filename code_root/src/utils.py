import argparse
import datetime as dt
import os
import sys

if "data_generation" in os.getcwd():
    sys.path.insert(1, "../code_root")
from Environment.enums import LogType
from os import getenv
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

EMAIL_ADDRESS = os.environ.get("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")

GMT5 = 18000
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
EARLY_PASSENGER_DELTA_MIN = 1


def convert_pandas_dow_to_pyspark(pandas_dow):
    return (pandas_dow + 1) % 7 + 1


def namespace_to_dict(namespace):
    return {k: namespace_to_dict(v) if isinstance(v, argparse.Namespace) else v for k, v in vars(namespace).items()}


def str_timestamp_to_datetime(timestamp_str):
    return dt.datetime.strptime(timestamp_str, DATETIME_FORMAT)


def str_timestamp_to_seconds(timestamp_str):
    return dt.datetime.strptime(timestamp_str, DATETIME_FORMAT).timestamp()


def seconds_epoch_to_str(timestamp_seconds):
    return dt.datetime.fromtimestamp(timestamp_seconds).strftime(DATETIME_FORMAT)


def datetime_to_str(_datetime):
    # return _datetime.strftime(DATETIME_FORMAT)
    return _datetime.strftime("%H:%M:%S")


def time_since_midnight_in_seconds(datetime_time):
    # t = dt.time(10, 10, 35)
    t = datetime_time
    td = dt.datetime.combine(dt.datetime.min, t) - dt.datetime.min
    seconds = td.total_seconds()  # Python 2.7+
    return seconds


def log(logger, curr_time=None, message=None, type=LogType.DEBUG):
    if logger is None:
        return

    if type == LogType.DEBUG:
        # self.logger.debug(f"[{seconds_epoch_to_str(curr_time)}] {message}")
        if curr_time:
            logger.debug(f"[{datetime_to_str(curr_time)}] {message}")
        else:
            logger.debug(f"{message}")
    if type == LogType.ERROR:
        # self.logger.error(f"[{seconds_epoch_to_str(curr_time)}] {message}")
        if curr_time:
            logger.error(f"[{datetime_to_str(curr_time)}] {message}")
        else:
            logger.debug(f"{message}")
    if type == LogType.INFO:
        # self.logger.info(f"[{seconds_epoch_to_str(curr_time)}] {message}")
        if curr_time:
            logger.info(f"[{datetime_to_str(curr_time)}] {message}")
        else:
            logger.debug(f"{message}")


def get_tod(timestamp):
    h = timestamp.hour
    if h < 6:
        return "early_am"
    elif h >= 6 and h < 9:
        return "rush_am"
    elif h >= 9 and h < 13:
        return "mid_am"
    elif h >= 13 and h < 17:
        return "mid_pm"
    elif h >= 17 and h < 19:
        return "rush_pm"
    elif h >= 20 and h < 24:
        return "night"
    else:
        return None


import os
import argparse
import tarfile
import json

# Import smtplib for the actual sending function.
import smtplib
from datetime import datetime

# Here are the email package modules we'll need.
from email.message import EmailMessage


def send_email(subject, message):
    smtpobj = smtplib.SMTP("smtp.gmail.com", 587)
    # start TLS for security which makes the connection more secure
    smtpobj.starttls()
    senderemail_id = EMAIL_ADDRESS
    senderemail_id_password = EMAIL_PASSWORD
    receiveremail_id = EMAIL_ADDRESS
    # Authentication for signing to gmail account
    try:
        smtpobj.login(senderemail_id, senderemail_id_password)
        # message to be sent
        # message = f"Finished running: {config['mcts_log_name']} on digital-storm-2."
        SUBJECT = subject
        message = "Subject: {}\n\n{}".format(SUBJECT, f"{message}.")
        smtpobj.sendmail(senderemail_id, receiveremail_id, message)
        # Hereby terminate the session
        smtpobj.quit()
        print("mail send - Using simple text message")
    except Exception as e:
        print(f"Sending mail error: {e}")
        print("Provide email address and password in .env file at git root to send results via email.")


def emailer(config_path=None, config_dict=None, msg_content=""):
    # Create the container email message.
    msg = EmailMessage()
    msg["Subject"] = "MCTS Stationing finished"
    me = EMAIL_ADDRESS
    recipients = [EMAIL_ADDRESS]
    msg["From"] = me
    msg.set_content(msg_content)

    try:
        msg["To"] = ", ".join(recipients)
        msg.preamble = "You will not see this in a MIME-aware mail reader.\n"

        # Open the files in binary mode.  You can also omit the subtype
        # if you want MIMEImage to guess it.

        now = datetime.now()
        log_name = now.strftime("%Y-%m-%d")

        if config_path and not config_dict:
            with open(config_path) as f:
                config = json.load(f)
        elif config_dict and not config_path:
            config = config_dict
            pass
        else:
            raise Exception("Config path or config dict must be provided.")

        output_tar_file = f"{config['mcts_log_name']}.tar.gz"

        res_dir = f"./results/{config['mcts_log_name']}"
        log_path = f"{res_dir}/results.csv"

        with tarfile.open(output_tar_file, "w:gz") as tar:
            tar.add(log_path, arcname=os.path.basename(log_path))

            if config_path and not config_dict:
                tar.add(config_path, arcname=os.path.basename(config_path))
            elif config_dict and not config_path:
                # Specify the file path where you want to save the JSON file
                file_path = f"{res_dir}/config.json"

                # Write the dictionary to the JSON file
                with open(file_path, "w") as json_file:
                    json.dump(config_dict, json_file)

                tar.add(file_path, arcname=os.path.basename(file_path))
            else:
                raise Exception("Config path or config dict must be provided.")

        for file in [output_tar_file]:
            filename = os.path.basename(os.path.normpath(file))
            with open(file, "rb") as fp:
                img_data = fp.read()
            msg.add_attachment(img_data, maintype="application", subtype="tar+gzip", filename=filename)

        # Send the email via our own SMTP server.
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            s.send_message(msg)
            s.quit()
    except Exception as e:
        print(f"Sending mail error: {e}")
        print("Provide email address and password in .env file at git root to send results via email.")
        pass
