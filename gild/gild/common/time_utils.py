import datetime
from datetime import timedelta

class time_zone(datetime.tzinfo):
    def __init__(self, name):
        # Options:
        # - US/Central
        self.name = name
    
    def utcoffset(self, dt):
        if self.name == 'US/Central':
            return timedelta(hours=-5)
        else:
            raise NotImplementedError
    
    def dst(self, dt:datetime.datetime):
        if self.name == 'US/Central':
            dston = datetime.datetime(year=dt.year, month=3, day=12)
            dstoff = datetime.datetime(year=dt.year, month=11, day=5)
        else:
            dston = datetime.datetime(year=dt.year, month=12, day=31)
            dstoff = datetime.datetime(year=dt.year, month=1, day=1)
        if dston <= dt.replace(tzinfo=None) < dstoff:
            return timedelta(hours=1)
        else:
            return timedelta(0)

def get_time_now(timezone='US/Central'):
    return datetime.datetime.now(tz=time_zone(timezone))

if __name__ == '__main__':
    print(get_time_now())
