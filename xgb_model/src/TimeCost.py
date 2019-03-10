import time


class TimeCost:

    def __init__(self):
        self.new_time = 0
        self.old_time = time.time()
        self.events = ['reading data',
                       'feature processing',
                       'splitting data',
                       'model training']
        self.count = 0

    def print_event(self):
        self.new_time = time.time()

        print(self.events[self.count] + str(' costs: ') + str(round(self.new_time - self.old_time, 2)) + 'sec')

        self.count += 1
        self.old_time = self.new_time

