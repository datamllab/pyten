import os
from pyten.UI import basic, auxiliary, dynamic
from Tkinter import *
from tkFileDialog import askopenfilename


class TensorUI:
    def __init__(self, parent):
        # Define basic methods and scenarios as lists and dictionaries
        self.scenarios = ["Basic Tensor", "With Auxiliary Info", "Online Tensor Completion"]
        self.methods = {"Basic Tensor": ["Tucker(ALS)", "CP(ALS)", "NNCP"], "With Auxiliary Info": ["AirCP", "CMTF"],
                        "Online Tensor Completion": ["OnlineCP", "OLSGD"]}
        self.methodsValue = {"Tucker(ALS)": '1', "CP(ALS)": '2', "NNCP": '3', "AirCP": '1', "CMTF": '2',
                             "OnlineCP": '2', "OLSGD": '3'}
        self.OPERATIONS = [("Recover", '1'), ("Decompose", '2')]
        self.file_path = ''
        self.selectedScenario = self.scenarios[0]
        self.selectedMethods = self.methods["Basic Tensor"]
        self.selectedMethod = "Tucker(ALS)"
        self.auxfiles = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.auxFilesPath = []
        self.auxbbuttons = {}
        self.auxbentries = {}
        self.auxlabels = []
        self.auxmodes = []
        self.auxabuttons = []
        self.globalrowcount = 0
        self.changingrowcount = 2

        # Define the top level container
        self.myContainer1 = Frame(parent)
        self.myContainer1.pack()

        # Label and Drop Down for Scenarios
        self.label1 = Label(self.myContainer1, text="Select Scenario")
        self.label1.grid(row=self.globalrowcount, column=0)
        var1 = StringVar()
        var1.set(self.scenarios[0])
        scenario_option = OptionMenu(self.myContainer1, var1, *self.scenarios, command=self.event_scenarioChange)
        scenario_option.grid(row=self.globalrowcount, column=1)
        self.globalrowcount += 1

        # Label and Dynamic Dropdown for Methods
        self.label2 = Label(self.myContainer1, text="Select Method")
        self.label2.grid(row=self.globalrowcount, column=0)
        var2 = StringVar()
        var2.set(self.selectedMethods[0])
        self.method_option = OptionMenu(self.myContainer1, var2, *self.selectedMethods, command=self.getSelectedMethod)
        self.method_option.grid(row=self.globalrowcount, column=1)
        self.globalrowcount += 1

        # Dynamic Auxillary UI
        self.mode_label = Label(self.myContainer1, text="Auxiliary Mode")
        self.auxmode = StringVar()
        self.aux_modeno = Entry(self.myContainer1, width=50, textvariable=self.auxmode)
        self.auxfilepath = Entry(self.myContainer1, width=50, textvariable=self.file_path)
        self.auxbbutton = Button(self.myContainer1, text="Browse")
        self.abutton = Button(self.myContainer1, text="Add Another Auxiliary File")
        self.auxbbutton.bind("<Button-1>", lambda event: self.browsecsv(event, TRUE))
        self.abutton.bind("<Button-1>", self.getNewAuxRow)
        self.auxbbuttons[self.auxbbutton] = self.changingrowcount
        self.auxbentries[self.changingrowcount] = self.auxfilepath
        self.auxmodes.append(self.aux_modeno)
        self.auxlabels.append(self.mode_label)
        self.auxabuttons.append(self.abutton)

        # self.aux_fileCount_container.bind('<Return>', self.getAuxModeFiles)
        self.auxiliaryUI(0)
        self.auxVisibility = 0

        # Label and select file option
        self.globalrowcount += 100
        self.label3 = Label(self.myContainer1, text="Select File")
        self.label3.grid(row=self.globalrowcount, column=0)
        self.entry = Entry(self.myContainer1, width=50, textvariable=self.file_path)
        self.entry.grid(row=self.globalrowcount, column=1)
        self.bbutton = Button(self.myContainer1, text="Browse")
        self.bbutton.bind("<Button-1>", self.browsecsv)
        self.bbutton.grid(row=self.globalrowcount, column=2, sticky='w')
        self.globalrowcount += 1

        # Operation Lable and Radio Buttons
        self.label4 = Label(self.myContainer1, text="Operation")
        self.label4.grid(row=self.globalrowcount, column=0)
        self.op = StringVar()
        self.op.set('1')
        for text, mode in self.OPERATIONS:
            b = Radiobutton(self.myContainer1, text=text, variable=self.op, value=mode)
            b.grid(row=self.globalrowcount, column=mode)
        self.globalrowcount += 1

        # Submit button
        self.button1 = Button(self.myContainer1, text="Submit", command=self.submitClick)
        self.button1.grid(row=self.globalrowcount, column=1)

    def auxiliaryUI(self, state):
        if state == 1:
            self.mode_label.grid(row=self.changingrowcount, column=0)
            self.aux_modeno.grid(row=self.changingrowcount, column=1)
            self.auxfilepath.grid(row=self.changingrowcount, column=2)
            self.auxbbutton.grid(row=self.changingrowcount, column=3)
            self.abutton.grid(row=self.changingrowcount, column=4)
        else:
            for i in self.auxlabels:
                i.grid_forget()
            for k, v in self.auxbentries.iteritems():
                v.grid_forget()
            for i in self.auxmodes:
                i.grid_forget()
            for k, v in self.auxbbuttons.iteritems():
                k.grid_forget()
            for i in self.auxabuttons:
                i.grid_forget()

    # Event Handler for Submit Button
    def submitClick(self):
        print self.auxFilesPath

        if self.selectedScenario == self.scenarios[0]:
            value = '1'
            basic(self.file_path, self.methodsValue[self.selectedMethod], self.op.get())
        elif self.selectedScenario == self.scenarios[1]:
            value = '2'
            auxiliary(self.file_path, self.methodsValue[self.selectedMethod], ' '.join(i.get() for i in self.auxmodes),
                      self.auxFilesPath, self.op.get())
        else:
            value = '3'
            dynamic(self.file_path, self.methodsValue[self.selectedMethod], self.op.get())

    # Event Handler for Changing Scenarios from Dropdown
    def event_scenarioChange(self, value):
        self.selectedScenario = value
        var2 = StringVar()
        if self.selectedScenario == self.scenarios[0]:
            self.selectedMethods = self.methods[self.scenarios[0]]
            self.auxiliaryUI(0)
        elif self.selectedScenario == self.scenarios[1]:
            self.selectedMethods = self.methods[self.scenarios[1]]
            self.auxiliaryUI(1)
        else:
            self.selectedMethods = self.methods[self.scenarios[2]]
            self.auxiliaryUI(0)

        var2.set(self.selectedMethods[0])
        self.method_option.grid_forget()
        self.method_option = OptionMenu(self.myContainer1, var2, *self.selectedMethods, command=self.getSelectedMethod)

        self.method_option.grid(row=1, column=1)

    def browsecsv(self, event, aux=FALSE):

        if not aux:
            Tk().withdraw()
            self.filename = askopenfilename()
            self.file_path = os.path.abspath(self.filename)
            self.entry.delete(0, END)
            self.entry.insert(0, self.file_path)
        else:
            b = event.widget
            Tk().withdraw()
            self.filename = askopenfilename()
            self.temppath = os.path.abspath(self.filename)
            self.auxbentries[self.auxbbuttons[b]].delete(0, END)
            self.auxbentries[self.auxbbuttons[b]].insert(0, self.temppath)
            self.auxFilesPath.append(self.temppath)

    def getSelectedMethod(self, value):
        self.selectedMethod = value

    def getNewAuxRow(self, event=None):
        for i in self.auxabuttons:
            i.grid_forget()
        self.changingrowcount += 1
        tempmode_label = Label(self.myContainer1, text="Auxiliary Mode")
        tempauxmode = StringVar()
        file_path = ''
        tempaux_modeno = Entry(self.myContainer1, width=50, textvariable=tempauxmode)
        auxfilepath = Entry(self.myContainer1, width=50, textvariable=file_path)
        auxbbutton = Button(self.myContainer1, text="Browse")
        tempmode_label.grid(row=self.changingrowcount, column=0)
        tempaux_modeno.grid(row=self.changingrowcount, column=1)
        auxfilepath.grid(row=self.changingrowcount, column=2)
        auxbbutton.grid(row=self.changingrowcount, column=3)
        self.auxbbuttons[auxbbutton] = self.changingrowcount
        self.auxbentries[self.changingrowcount] = auxfilepath
        auxbbutton.bind("<Button-1>", lambda event: self.browsecsv(event, TRUE))
        abutton = Button(self.myContainer1, text="Add Another Auxiliary File")
        abutton.grid(row=self.changingrowcount, column=4)
        abutton.bind("<Button-1>", self.getNewAuxRow)
        self.auxmodes.append(tempaux_modeno)
        self.auxlabels.append(tempmode_label)
        self.auxabuttons.append(abutton)


root = Tk()
root.wm_title("PyTen")
myapp = TensorUI(root)
root.mainloop()
