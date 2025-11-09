import re

with open('heytea.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 替换所有 cv2.drawContours 为抗锯齿版本
pattern = r'cv2\.drawContours\(preview_canvas, final_contours, -1, \(0, 0, 255\), int\(preview_thickness\)\)'
replacement = '''for contour in final_contours:
            cv2.polylines(preview_canvas, [contour], False, (0, 0, 255), int(preview_thickness), lineType=cv2.LINE_AA)'''

content = re.sub(pattern, replacement, content)

with open('heytea.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("抗锯齿补丁已应用")
