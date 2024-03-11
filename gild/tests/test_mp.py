import multiprocessing

class MyProcess(multiprocessing.Process):
    def __init__(self, arg):
        super(MyProcess, self).__init__()
        self.arg = arg

    def run(self):
        print(f"Process running with argument: {self.arg}")
        # Place the code here that you want the process to execute

class parentProcess:
    def __init__(self):
        self.childs = [MyProcess(f'hello {_}') for _ in range(5)]

    def start(self):
        for child in self.childs:
            child.start()

# # Creating and starting the process
# process = MyProcess("Hello World")
# process.start()

# # Wait for the process to complete (optional)
# process.join()

process = parentProcess()
process.start()
