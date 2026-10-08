from dataclasses import dataclass # dataclasses acts as a decorator, creates variable for an empty class 

@dataclass
class DataIngestionArtifact:
    trained_file_path:str
    test_file_path:str