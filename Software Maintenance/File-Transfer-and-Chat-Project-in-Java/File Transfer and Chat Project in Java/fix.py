import os
import javalang

# CHANGE ONLY THIS LINE — your real project folder
source_dir = r"D:\8th\Software Maintenance\File-Transfer-and-Chat-Project-in-Java\File Transfer and Chat Project in Java"

output_file = os.path.join(source_dir, "extra_correct.csv")

with open(output_file, "w", encoding="utf-8") as out:
    out.write("class,cl_comf,cl_stat\n")   # header

    total_comf = 0.0
    total_stat = 0

    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".java"):
                path = os.path.join(root, file)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        lines = f.readlines()

                    # count comment lines
                    comment_lines = 0
                    in_multi = False
                    for line in lines:
                        s = line.strip()
                        if in_multi:
                            comment_lines += 1
                            if "*/" in s: in_multi = False
                            continue
                        if s.startswith("//"):
                            comment_lines += 1
                        elif "/*" in s:
                            comment_lines += 1
                            if "*/" not in s: in_multi = True

                    total_lines = sum(1 for line in lines if line.strip())
                    comf = comment_lines / total_lines if total_lines > 0 else 0

                    # count statements
                    stmts = 0
                    with open(path, "r", encoding="utf-8") as f:
                        tree = javalang.parse.parse(f.read())
                    for _, node in tree.filter(javalang.tree.Statement):
                        stmts += 1

                    class_name = file.replace(".java", "")
                    out.write(f"{class_name},{round(comf,6)},{stmts}\n")

                    total_comf += comf
                    total_stat += stmts

                except:
                    pass   # skip files that can't be parsed

    # add the totals at the end
    out.write(f"\nTOTAL_COMF,,{round(total_comf,6)}\n")
    out.write(f"TOTAL_STAT,,{total_stat}\n")

print("Done! File created → extra_correct.csv")
print(f"cl_comf total = {round(total_comf,6)}")
print(f"cl_stat total = {total_stat}")