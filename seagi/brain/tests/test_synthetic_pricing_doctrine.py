"""SYNTHETIC PRICING DOCTRINE — every synthetic=True site must be priced.

`synthetic=True` is ECONOMY-CRITICAL.  The moment a concept is minted
synthetic, MortalityClock._fact_key collapses:
  * every membership edge into it  -> the single fact ('∈', name)
  * members' machinery-entailed derived edges -> (name, r, t)
so N future member settlings become 1.  A new synthetic-minting site
added without thought silently changes what SEAGI can earn from.

This test AST-scans all of `seagi/` for synthetic=True construction and
attribute-assignment sites and asserts the discovered set equals the
declared registry below.  Adding a minting site is allowed — adding one
WITHOUT registering it and asserting its pricing is not.

Registry reconciled 2026-07-19 (step 0).  Note for readers coming from
the design doc: the audit cited "substrate.py:2448 and :2460-2468" as a
pair.  Only 2448 is a construction site; 2460-2468 is the membership
is_a EDGE write, not a Concept(synthetic=True) call.
"""

from __future__ import annotations

import ast
import os
import unittest

_SEAGI_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

# (relative path from the seagi package root, enclosing function)
DECLARED_SYNTHETIC_SITES = {
    ('core/substrate.py', 'form_abstractions'),
    ('core/substrate.py', '_form_abstractions_scoped'),
}

_DOCTRINE = (
    "\n\nSYNTHETIC PRICING DOCTRINE: `synthetic=True` is economy-critical "
    "-- membership edges collapse to family grain in "
    "MortalityClock._fact_key (a class's memberships become the single "
    "fact ('∈', class), and members' entailed derivations become "
    "(class, r, t)).  Register the site in DECLARED_SYNTHETIC_SITES and "
    "add a pricing assertion that states what the new class collapses.\n")


def _enclosing_functions(tree):
    parent_fn = {}

    def walk(node, fn):
        for child in ast.iter_child_nodes(node):
            newfn = fn
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                newfn = child.name
            parent_fn[child] = newfn
            walk(child, newfn)
    walk(tree, None)
    return parent_fn


def _scan():
    """-> (set of (relpath, funcname), list of (relpath, lineno, fn, kind))"""
    found = set()
    detail = []
    for dirpath, dirnames, filenames in os.walk(_SEAGI_ROOT):
        dirnames[:] = [d for d in dirnames if d != '__pycache__']
        for fn in sorted(filenames):
            if not fn.endswith('.py'):
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, _SEAGI_ROOT).replace(os.sep, '/')
            try:
                with open(full, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read(), filename=rel)
            except (OSError, SyntaxError):
                continue
            pfn = _enclosing_functions(tree)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    for kw in node.keywords:
                        if (kw.arg == 'synthetic'
                                and isinstance(kw.value, ast.Constant)
                                and kw.value.value is True):
                            found.add((rel, pfn.get(node)))
                            detail.append(
                                (rel, node.lineno, pfn.get(node), 'call'))
                if isinstance(node, ast.Assign):
                    for tgt in node.targets:
                        if (isinstance(tgt, ast.Attribute)
                                and tgt.attr == 'synthetic'
                                and isinstance(node.value, ast.Constant)
                                and node.value.value is True):
                            found.add((rel, pfn.get(node)))
                            detail.append(
                                (rel, node.lineno, pfn.get(node), 'assign'))
    return found, detail


class TestSyntheticPricingDoctrine(unittest.TestCase):

    def test_site_set_matches_declared_registry(self):
        found, detail = _scan()
        # exclude the test tree itself: fixtures may mint freely
        found = {(r, f) for (r, f) in found
                 if '/tests/' not in ('/' + r)}
        unregistered = found - DECLARED_SYNTHETIC_SITES
        missing = DECLARED_SYNTHETIC_SITES - found
        msg = ''
        if unregistered:
            msg += ('\nUNREGISTERED synthetic=True site(s): %s'
                    % sorted('%s::%s' % s for s in unregistered))
        if missing:
            msg += ('\nDECLARED site(s) no longer present (registry '
                    'stale, or the minting moved): %s'
                    % sorted('%s::%s' % s for s in missing))
        if msg:
            msg += '\n\nAll sites discovered:\n' + '\n'.join(
                '  %s:%d in %s (%s)' % d for d in sorted(detail))
            msg += _DOCTRINE
        self.assertEqual(found, DECLARED_SYNTHETIC_SITES, msg)

    def test_both_declared_sites_mint_abstraction_classes(self):
        """PRICING ASSERTION for the two registered sites: both mint an
        `_abstract_{relation}_{target}` class whose memberships the clock
        then prices as ONE fact."""
        from seagi.core.substrate import (
            Substrate, ABSTRACTION_NAME_PREFIX, ABSTRACTION_RELATION)
        from seagi.brain.capabilities.mortality_clock import (
            MortalityClock, MEMBERSHIP_KEY_HEAD)

        class _Drive:
            deaths = 0
            revivals = 0
            wall = 0.1
            obstruction_ema = 0.0
            _last_wall_advance = 0.0

        for scoped in (False, True):
            sub = Substrate()
            for m in ('robin', 'crow', 'wren'):
                sub.add_edge(source=m, target='feathered',
                             relation_name='has_property',
                             strength=0.5, cycle=0)
            if scoped:
                sub.form_abstractions(
                    cycle=1, min_group=3,
                    candidates={'robin', 'crow', 'wren'})
            else:
                sub.form_abstractions(cycle=1, min_group=3)
            cls = ABSTRACTION_NAME_PREFIX + 'has_property_feathered'
            self.assertIn(cls, sub.concepts,
                          'scoped=%s should mint the class' % scoped)
            self.assertTrue(sub.concepts[cls].synthetic)
            clock = MortalityClock(
                mortality_drive=_Drive(),
                substrate_provider=lambda s=sub: s,
                log_path='', death_log_path='')
            try:
                expected = clock._khash((MEMBERSHIP_KEY_HEAD, cls))
                seen = set()
                for m in ('robin', 'crow', 'wren'):
                    e = sub.edges.get((m, ABSTRACTION_RELATION, cls))
                    if e is not None:
                        seen.add(clock._fact_key(e))
                self.assertTrue(seen, 'expected membership edges')
                self.assertEqual(
                    seen, {expected},
                    'every membership of a minted class must price as the '
                    'single family fact' + _DOCTRINE)
            finally:
                from seagi.core.substrate import (
                    unregister_reinforce_observer)
                unregister_reinforce_observer()


if __name__ == '__main__':
    unittest.main()
