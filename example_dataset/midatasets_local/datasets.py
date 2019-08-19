from midatasets.MIReader import MIReader


class LiverReader(MIReader):
    name = 'liver'

    def __init__(self, spacing, do_preprocessing=False):

        self.dir_path = self.get_root_path() + '/CT/liver'
        super().__init__(spacing, dir_path=self.dir_path)

        self.labels = [0, 1, 2]
        self.label_names = ['background', 'liver', 'lesion']
        self.spacing = spacing
        self.do_preprocessing = do_preprocessing


class LungReader(MIReader):


    def __init__(self, spacing, do_preprocessing=False):
        self.dir_path = self.get_root_path() + '/CT/lung'
        super().__init__(spacing, dir_path=self.dir_path)

        self.labels = [0, 1]
        self.label_names = ['background', 'lesion']
        self.spacing = spacing
        self.do_preprocessing = do_preprocessing

