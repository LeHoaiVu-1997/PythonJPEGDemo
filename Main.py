import os
import DCT


path1 = "D:/Python projects/JPEGCompression/JPG Images"
for img in os.listdir(path1):
    imageName = os.path.basename(os.path.join(path1, img))
    print(imageName)
    DCT.CompressNDecompress(imageName, 10, "JPG")
    DCT.CompressNDecompress(imageName, 50, "JPG")
    DCT.CompressNDecompress(imageName, 75, "JPG")
    DCT.CompressNDecompress(imageName, 100, "JPG")
    print("================== Finish a pic ==============================")


path2 = "D:/Python projects/JPEGCompression/PNG Images"
for img in os.listdir(path2):
    imageName = os.path.basename(os.path.join(path2, img))
    print(imageName)
    DCT.CompressNDecompress(imageName, 10, "PNG")
    DCT.CompressNDecompress(imageName, 50, "PNG")
    DCT.CompressNDecompress(imageName, 75, "PNG")
    DCT.CompressNDecompress(imageName, 100, "PNG")
    print("================== Finish a pic ==============================")