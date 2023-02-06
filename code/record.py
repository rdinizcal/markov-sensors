class Record : 

    def __init__ (self, recordID, age, gender, height, icutype, weight):
        self.recordID = recordID
        self.age = age
        self.gender = gender
        self.height = height
        self.weight = weight

        icutype = int(icutype)
        if (icutype == 1):
            self.icutype = "Coronary Care Unit"
        elif (icutype == 2):
            self.icutype = "Cardiac Surgery Unit"
        elif (icutype == 3):
            self.icutype = "Medical ICU"
        elif (icutype == 4):
            self.icutype = "Surgical ICU"
        else :
            self.icutype = "unknown"
        
        self.vital_signals = {str: []}
