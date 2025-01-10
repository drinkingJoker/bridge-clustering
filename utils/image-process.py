from PIL import Image, ImageEnhance

# 打开图像文件
image = Image.open("data/images/1.png")

# 创建增强器对象
enhancer = ImageEnhance.Sharpness(image)

# 应用锐化效果，factor > 1 增加锐度
sharpened_image = enhancer.enhance(2.0)

# 保存或显示结果
sharpened_image.save("sharpened_image.png")
sharpened_image.show()
