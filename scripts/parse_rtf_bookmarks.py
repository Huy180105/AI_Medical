import sys, re
sys.stdout.reconfigure(encoding='utf-8')

path = r'C:\Users\Quang Huy\Downloads\Báo cáo AI_SamSung.rtf'
with open(path, 'rb') as f:
    raw = f.read()

def decode_text(b):
    text = b.decode('latin1', errors='ignore')
    def u_rep(m):
        c = int(m.group(1))
        return chr(c if c >= 0 else c + 65536)
    text = re.sub(r'\\u([0-9\-]+)\??', u_rep, text)
    def h_rep(m):
        try:
            return bytes.fromhex(m.group(1)).decode('windows-1258')
        except:
            return ''
    text = re.sub(r'\\\'([0-9a-fA-F]{2})', h_rep, text)
    text = re.sub(r'\\[a-zA-Z0-9\-]+ ?', ' ', text)
    text = re.sub(r'[{}]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Find all bookmarks
# pattern: {\*\bkmkstart (_Toc\d+)} (anything) {\*\bkmkend \1}
pattern = re.compile(rb'\\bkmkstart (_Toc\d+)\}(.*?)\\bkmkend \1', re.DOTALL)
matches = pattern.findall(raw)
print(f'Total headings found via bookmarks: {len(matches)}')

headings = []
for tag, content in matches:
    title = decode_text(content)
    if title:
        headings.append((tag.decode(), title))
        print(f'{tag.decode()}: {title}')

with open('headings_extracted.txt', 'w', encoding='utf-8') as f:
    for tag, title in headings:
        f.write(f'{tag}: {title}\n')
