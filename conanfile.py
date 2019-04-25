from conans import ConanFile

class MyConan(ConanFile):
    settings = 'os', 'compiler', 'build_type', 'arch'
    generators = 'cmake'


    def requirements(self):
        self.requires('gflags/2.2.2@bincrafters/stable')

    def configure(self):
        self.options['gflags'].shared = True
        self.options['gflags'].nothreads = True
