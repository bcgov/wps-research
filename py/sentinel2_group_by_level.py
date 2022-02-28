'''make two folders and group Sentinel2 dates by Level 1, and Level 2
20220228'''

from misc import run

run('mkdir -p L1C')
run('mkdir -p L2A')
run('mv -v *L1C* L1C')
run('mv -v *L2A* L2A')
