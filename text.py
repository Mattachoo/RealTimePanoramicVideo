class PriorityQueue(object):
    def __init__(self):
        self.queue = []

    def __str__(self):
        return ' '.join([str(i) for i in self.queue])

    # for checking if the queue is empty
    def isEmpty(self):
        return len(self.queue) == 0

    # for inserting an element in the queue
    def insert(self, data):
        max_val = -1
        if len(self.queue) < 1:
            self.queue.append(data)
            return
        for i in range(len(self.queue)):
            if self.queue[i][0] > data[0]:
                self.queue.insert(i, data)
                return
        self.queue.append(data)

    # for popping an element based on Priority
    def pop(self):
        return self.queue.pop(0)
    def deleted(self, index):
        return self.queue.pop(index)
myQueue = PriorityQueue()
myQueue.insert((12, " World"))
myQueue.insert((1, "Hello"))
myQueue.insert((14, "!"))
myQueue.insert((7," "))
myQueue.pop()
print(myQueue)