import os

current_path = os.getcwd()

directories = [
    os.path.join(".github", "workflows"),
    "artifacts",
    os.path.join("data", "raw"),
    os.path.join("data", "processed"),
    "data_given",
    "notebooks",
    os.path.join("prediction_service", "model"),
    "report",
    "saved_models",
    "src",
    "tests",
    os.path.join("web_app", "static"),
    os.path.join("web_app", "templates"),
    os.path.join("web_app/static", "css"),
    os.path.join("web_app/static", "script"),
]

files = [
    ".gitignore",
    "Procfile"
    "README.md",
    "LICENSE",
    "app.py",
    "dvc.yaml",
    "params.yaml",
    "requirements.txt",
    "setup.py",
    "tox.ini",
    os.path.join('src', '__init__.py')
]


class DsWorkflowCreator:
    def __init__(self, directories, files):
        self.directories = directories
        self.files = files

    def _create_directories(self):

        for dir in self.directories:
            directory = os.path.join(current_path, dir)
            os.makedirs(directory, exist_ok=True)

            with open(os.path.join(directory, ".gitkeep"), "w") as f:
                pass

    def _create_files(self):
        for file in self.files:
            make_file = os.path.join(current_path, file)
            with open(os.path.join(make_file), "w") as f:
                pass

    def ds_workflow(self):
        self._create_directories()
        self._create_files()


def create_my_ds_workflow(directories=directories, files=files):
    ds_workflow_creator = DsWorkflowCreator(directories, files)
    return ds_workflow_creator.ds_workflow()


if __name__ == '__main__':
    create_my_ds_workflow()