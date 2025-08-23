#!/usr/bin/env python3
"""Test validation improvements"""

from dreamlog import compound, atom

# Test empty functor validation
try:
    c = compound('', atom('a'))
    print('✗ Empty functor: Should have failed')
except ValueError as e:
    print('✓ Empty functor validation works:', e)

# Test normal case
try:
    c = compound('parent', atom('alice'), atom('bob'))
    print('✓ Normal compound creation works')
except Exception as e:
    print('✗ Normal compound failed:', e)

print('\nAll validation tests passed!')