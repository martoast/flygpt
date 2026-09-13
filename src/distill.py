"""Use the matched language runner with --teacher, --alpha and --temperature."""
import sys
from .train_language import main

if __name__=='__main__':
    if '--teacher' not in sys.argv:raise SystemExit('Distillation requires --teacher CHECKPOINT')
    main()
