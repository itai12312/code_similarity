from abc import ABCMeta, abstractmethod
import time
import os
from multiprocessing import Process

RETRIES = 15
SLEEP = 1
TIME_PAUSE_BETWEEN = 0.01

class BaseTask():
    __metaclass__ = ABCMeta

    def __init__(self, in_queue=None, out_queue=None, bulksize=1):
        self.in_queue = in_queue
        self.out_queue = out_queue
        #self.inbulk = inbulk
        self.cur_time = time.time()
        self.items_count = 0
        self.prev_time = -1
        self.start_time = time.time()
        self.bulksize = bulksize

    def run_tasks_from_queue(self):
        assert self.in_queue
        # continu = True
        # while continu:
            # for tries in range(RETRIES):
        time.sleep(SLEEP) # to prevent thread reset by user errors in rabbitmq
        while not self.in_queue.empty():
            item = self.in_queue.get()
            time.sleep(TIME_PAUSE_BETWEEN)
            self._run_task(item) # status =
            # self.in_queue.send_ack()
        #if not continueonempty and self.in_queue.empty():
        #    cont = False
        #    break
        time.sleep(SLEEP)
    @abstractmethod
    def task(self, item):
        pass

    @classmethod
    def enqueque_items(cls, queque, items):
        if queque is not None and items is not None:
            for item in items:
                queque.put(item)

    def _run_task(self, item):
        out_items = self.task(item)
        # BaseTask.enqueque_items(self.out_queue, out_items)

class TimedProcess:
    def __init__(self, process, time):
        self.process = process
        self.time = time

def spawn(task, number):
    timed_processes = []
    for i in range(number):
        timed_processes.append(
            TimedProcess(Process(target=task), time.time()))  # make sure comma after args is sending args
        #timed_processes[i].deamon = True
        timed_processes[i].process.start()
    return timed_processes


def join(timed_processes):
    for tp in timed_processes:
        tp.process.join()  # Wait for the reader to finish
        print("Sending numbers to Queue() took %s seconds" % ((time.time() - tp.time)))


def multi_process_run(task, count):
    timed_processes = spawn(task.run_tasks_from_queue, count)
    join(timed_processes)
