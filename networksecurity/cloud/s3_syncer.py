'''
Helper to copy folders between the local machine and an AWS S3 bucket.
Uses the AWS CLI, so it must be installed and set up with "aws configure".
'''

import subprocess


class S3Sync:
    '''Upload or download a whole folder with "aws s3 sync" (only new or changed files are copied).'''

    def sync_folder_to_s3(self, folder, aws_bucket_url):
        '''Upload a local folder to S3, e.g. Artifacts/<timestamp> -> s3://bucket/artifact/<timestamp>.'''
        command = f"aws s3 sync {folder} {aws_bucket_url} "
        subprocess.run(command)  # runs the CLI command; note it does not raise an error if the upload fails

    def sync_folder_from_s3(self, folder, aws_bucket_url):
        '''Download a folder from S3 to local, e.g. to fetch the final model before prediction.'''
        command = f"aws s3 sync {aws_bucket_url} {folder} "
        subprocess.run(command)