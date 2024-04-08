with open('data.txt', 'r', encoding="utf-8") as f:
    lines = f.readlines()
    new_lines = []
    for i in range(len(lines)):
        if i == 0:
            new_lines.append(lines[i])
        elif not lines[i][:10].isdigit():
            new_lines[-1] = new_lines[-1].strip() + lines[i]
        else:
            new_lines.append(lines[i])

new_lines = [line for line in new_lines if line.strip()]

with open('PHEME2.txt', 'w', encoding="utf-8") as f:
    for item in new_lines:
        f.write("%s\n" % item)
