import os.path
import threading
import time

import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage

key_path = os.path.join(os.getcwd(), 'wardrobeServiceAccoutKey.json')
cred = credentials.Certificate(key_path)
firebase_admin.initialize_app(cred, {
    'storageBucket': 'capstone-wardrobe.appspot.com'
})


def upload_blob_from_memory_task(contents, destination_blob_name, mime_type) -> tuple[str, threading.Thread]:
    """Uploads a file to the bucket."""

    # The contents to upload to the file
    # contents = "these are my contents"

    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"

    bucket = storage.bucket()
    blob = bucket.blob(destination_blob_name)
    thread = threading.Thread(target=blob.upload_from_string, args=(contents, mime_type))
    thread.start()

    expire_epoch = int(time.time()) + 3600
    return blob.generate_signed_url(expiration=expire_epoch), thread
