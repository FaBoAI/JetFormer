"""台本 1 の 2 つの形(spin・ceil と hit・left42)の ⑤ の標準出力の全部を out/ に写す(人が読む行の並びを見るため。読むだけ)

  jf-heavy --lock none --mem 1500M -- python -B show_t1_stdout.py
"""
import common as C
import probe_t1_trial_form as P


def main():
    for shape, form in (('spin', 'ceil'), ('hit', 'left42')):
        D, log = P.day(C.fresh('s1_%s_%s' % (shape, form)), shape, form)
        j, out, _ = C.runc(D, 'new', C.IG, 's1%s%s' % (shape, form))
        (C.OUT / ('probe_t1_stdout_%s_%s.txt' % (shape, form))).write_text(C.t7.no_path(out) + '\n', encoding='utf-8')
        print('==', shape, form)
        print(C.t7.no_path(out))
    C.cleanup()


if __name__ == '__main__':
    main()
