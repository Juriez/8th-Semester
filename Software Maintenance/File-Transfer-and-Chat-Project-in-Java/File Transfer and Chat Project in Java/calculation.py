import os
import javalang
import pandas as pd

source_dir = r"D:\8th\Software Maintenance\File-Transfer-and-Chat-Project-in-Java\File Transfer and Chat Project in Java"  # Example: C:\...\\File Transfer and Chat Project in Java

data = []
for root, _, files in os.walk(source_dir):
    for file in files:
        if file.endswith('.java'):
            path = os.path.join(root, file)
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            total_lines = sum(1 for line in lines if line.strip())
            comment_lines = 0
            in_multi = False
            for line in lines:
                s = line.strip()
                if in_multi:
                    comment_lines += 1
                    if '*/' in s:
                        in_multi = False
                    continue
                if s.startswith('//'):
                    comment_lines += 1
                elif '/*' in s and '*/' in s:
                    comment_lines += 1
                elif '/*' in s:
                    comment_lines += 1
                    in_multi = True
            
            comf = comment_lines / total_lines if total_lines > 0 else 0
            
            stmts = 0
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    code = f.read()
                tree = javalang.parse.parse(code)
                for _, node in tree.filter(javalang.tree.Statement):
                    stmts += 1
            except:
                stmts = 0
            
            class_name = file.replace('.java', '')
            data.append({'class': class_name, 'cl_comf': round(comf, 4), 'cl_stat': stmts})

pd.DataFrame(data).to_csv(os.path.join(source_dir, 'extra.csv'), index=False)
print("extra.csv created!")