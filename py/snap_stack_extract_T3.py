from misc import run, err, exists, hdr_fn, args, band_names

if len(args) < 2:
    err("ts_stack_change.py [stack.bin produced from SNAP stack operator, followed by snap2psp.py ./ 1")

fn = args[1]
hfn = hdr_fn(fn)
bn = band_names(hfn)

phone_book = {}

lookup = {bn[i]: i for i in range(len(bn))}

for b in bn:
    w = b.split('_')

    ti = w[0].split('T')[1]

    if ti[0] == '4' or ti[1] == '4':
        continue  # use T3 elements only
    if not w[-1] in phone_book:
        phone_book[w[-1]] = []
    phone_book[w[-1]] += [b]

print(phone_book)
