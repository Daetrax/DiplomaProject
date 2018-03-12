import numpy as np
import os
from cntk.io import UserDeserializer, StreamInformation
import helper
import cv2


class MyDeserializer(UserDeserializer):

    def __init__(self, directory, streams, chunksize=32 * 1024 * 1024):
        super(MyDeserializer, self).__init__()
        self._chunksize = chunksize
        self._directory = directory
        self._dictionary = {}

        self._streams = [StreamInformation(s['name'], i, 'dense', np.float32, s['shape'])
                         for i, s in enumerate(streams)]

        # Define the number of chunks based on the file size
        self._num_chunks = int(
            len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))]) / 2)

        self.createDictionary(directory)

    def stream_infos(self):
        return self._streams

    def num_chunks(self):
        return self._num_chunks

    # Ok, let's actually get the work done
    def get_chunk(self, chunk_id):

        filePath = self._directory + self._dictionary[chunk_id]
        mask = filePath.replace(".tif", "")
        mask += "_mask.tif"

        # print("Getting chunk: ", chunk_id, "\nFilename: ", filePath)

        features, labels = helper.slide(cv2.imread(filePath), cv2.imread(mask))

        return {
            "features": np.array(features, np.float32),
            "labels": np.array(labels, np.float32)
        }

    def createDictionary(self, directory):
        counter = 0

        for name in os.listdir(directory):
            if "mask" in name:
                continue

            self._dictionary[counter] = name
            counter += 1
