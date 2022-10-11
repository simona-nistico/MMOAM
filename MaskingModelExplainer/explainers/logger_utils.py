import datetime


def time_format(i):
    """ takes a timestamp (seconds since epoch) and transforms that into a datetime string representation """
    return datetime.datetime.fromtimestamp(i).strftime('%Y%m%d%H%M%S')
