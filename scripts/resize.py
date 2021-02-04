from PIL import Image
import os.path
import glob
def convertjpg(jpgfile,outdir,width=256,height=256):
  img=Image.open(jpgfile)
  try:
    new_img=img.resize((width,height),Image.BILINEAR)  
    new_img.save(os.path.join(outdir,os.path.basename(jpgfile)))
  except Exception as e:
    print(e)
for jpgfile in glob.glob("./UVB_DRIT_fake/*.png"):
  convertjpg(jpgfile,"./UVB_DRIT_fake_256")
