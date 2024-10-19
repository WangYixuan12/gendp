import datetime

def get_current_YYYY_MM_DD_hh_mm_ss_ms():
    """
    Returns a string identifying the current:
    - year, month, day, hour, minute, second

    Using this format:

    YYYY-MM-DD-hh-mm-ss

    For example:

    2018-04-07-19-02-50

    Note: this function will always return strings of the same length.

    :return: current time formatted as a string
    :rtype: string

    """

    now = datetime.datetime.now()
    string =  "%0.4d-%0.2d-%0.2d-%0.2d-%0.2d-%0.2d-%0.6d" % (now.year, now.month, now.day, now.hour, now.minute, now.second, now.microsecond)
    return string
