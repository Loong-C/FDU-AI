from PIL import Image

frames = []

# for i in range(8):
#     img = Image.open(f"/Users/liuqingyun/Desktop/DISC/人工智能-25春/PJ/PJ1-搜索/pj1-search/images/Coins/coin_0{i+1}.png")
#     frames.append(img)

# frames[0].save(
#     "/Users/liuqingyun/Desktop/DISC/人工智能-25春/PJ/PJ1-搜索/pj1-search/images/coin.gif",
#     save_all=True,
#     append_images=frames[1:],
#     duration=120,  # 每帧120ms
#     loop=0,
#     disposal=2  # 每帧前清除前一帧
# )

img = Image.open("/Users/liuqingyun/Desktop/DISC/人工智能-25春/PJ/PJ1-搜索/pj1-search/images/DRagon/NES Dragon Sprite Sheet.png")
width = img.width//8
height = img.height//8
for i in range(4):
    frame = img.crop((i*width, height*7, (i+1)*width, height*8))
    frames.append(frame)
frames[0].save(
    "/Users/liuqingyun/Desktop/DISC/人工智能-25春/PJ/PJ1-搜索/pj1-search/images/DRagon/dragon_right_2.gif",
    save_all=True,
    append_images=frames[1:],
    duration=120,
    loop=0,
    disposal=2
)