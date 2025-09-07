import py_compile
import glob,sys
files=glob.glob('**/*.py',recursive=True)
failed=False
for f in files:
    try:
        py_compile.compile(f, doraise=True)
    except Exception as e:
        print('COMPILE FAIL',f, e)
        failed=True
if not failed:
    print('COMPILE_OK')
else:
    sys.exit(2)
