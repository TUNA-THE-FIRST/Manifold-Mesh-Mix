# Licence and origins of Mesh Mix-S2C

Mesh Mix-S2C merges two programs into one. **As a whole it is licensed under the GNU
Affero General Public License, version 3** (the text is in `LICENSE`). If you give it, or
a changed version of it, to anyone, or let others use a changed version over a network,
you must offer them its complete source under the same licence.

It is made of:

- **Manifold Mesh Mix**, Copyright (c) 2026 Tuna-The-First and Claude (Anthropic's AI
  assistant), AGPL-3.0: the apps (`app/`, `desktop/`) and the conversion engine in
  `engine/mmm/`, taken as published at https://github.com/TUNA-THE-FIRST/Manifold-Mesh-Mix
  (this repository continues it). Files of `engine/mmm/` that the merge changed say so at
  their top: what changed, and when. The software it depends on is listed in
  `THIRD_PARTY_NOTICES.md`.
- **stl2curves**, Copyright (c) 2026 Duvdivan, published under the MIT licence
  (https://github.com/Duvdivan/stl2curves; its text is below, and still applies to the
  parts of this repository that come from there as published). Everything in
  `engine/mms2c/` comes from there, except `merged.py`, `rivals.py` and `judge.py`,
  which were written for the merged engine. The package was called `stl2curves` there and
  was renamed `mms2c` here (its settings `STL2CURVES_*` became `MM_S2C_*`); the merged
  engine as a whole was brought in from the organization's merge repository at its commit
  ade0b31 (2026-10-31).

## The MIT licence of stl2curves

MIT License

Copyright (c) 2026 Duvdivan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
