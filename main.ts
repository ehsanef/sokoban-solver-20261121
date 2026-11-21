// Room-by-room push-optimal Sokoban solver, written for make.fun's
// sokoban-solver-01-run-20261121 bounty. One run answers every level in
// /input/<id>/level.txt (XSB) with /output/<id>/solution.txt, the player's
// moves in LURD.
//
// How it works:
//  - Every level here is a one-wide hallway with rooms off it, and a box can
//    never leave its room (a push back through a door needs the player on the
//    hallway's far side, which is wall), so each room is solved on its own
//    and the rooms' push-optimal counts add up to the level's optimum.
//  - A room is solved by A* over pushes: a state is the boxes' squares plus
//    the player's reachable region (named by its lowest square), each push
//    costs 1, and the estimate is the cheapest assignment of boxes to goals
//    under precomputed push-distance tables (admissible, so the first solved
//    state popped is push-optimal). Dead squares (no pull-back path to any
//    goal) and frozen 2x2 blocks of walls and boxes prune dead states.
//  - If the optimal search runs out of work budget on a room, a greedy
//    best-first search falls back to any solution for it: a solved level a
//    push over optimal still scores about half, an unsolved one scores 0.
//  - Everything is budgeted by work (node expansions), never by the clock:
//    the run is deterministic, and two executions write the same bytes.
// Local run: mkdir -p out && MAKEFUN_INPUT_DIR=practice/input MAKEFUN_OUTPUT_DIR=out node main.ts
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Work budgets in node expansions, never the clock: the run must be
// deterministic, and a wall-clock cut-off would answer differently per run.
const OPTIMAL_CAP = 300000; // expansions per room for the optimal A* (above the generator's own 250k-state room guarantee, so the optimal search cannot run out on a kept room)
const GREEDY_CAP = 200000; // expansions per room for the greedy fallback
const WHOLE_CAP = 400000; // expansions for a whole-level fallback search
const GLOBAL_CAP = 2500000; // expansions across the whole run, all levels

const DX = [0, 0, -1, 1];
const DY = [-1, 1, 0, 0];
const STEP = ['u', 'd', 'l', 'r'];
const PUSH = ['U', 'D', 'L', 'R'];
const INF = 1e9;

/** A parsed level: width, height, walls, goals, boxes and the player. */
function parseLevel(text) {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const size = w * h;
  const wall = new Uint8Array(size).fill(1);
  const goal = new Uint8Array(size);
  const goals = [];
  const boxes = [];
  let player = -1;
  for (let y = 0; y < h; y++) {
    const row = rows[y];
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') {
        goal[i] = 1;
        goals.push(i);
      }
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  }
  // The outermost frame is wall whatever it shows: nothing leaves the board.
  for (let x = 0; x < w; x++) {
    wall[x] = 1;
    wall[(h - 1) * w + x] = 1;
  }
  for (let y = 0; y < h; y++) {
    wall[y * w] = 1;
    wall[y * w + w - 1] = 1;
  }
  boxes.sort((a, b) => a - b);
  return { w, h, size, wall, goal, goals, boxes, player };
}

/**
 * Push-distance tables, one per goal: table[g][s] is the fewest pushes that
 * bring a lone box from square s onto goal g, ignoring the other boxes (INF
 * when no pushes do). A breadth-first search backwards from the goal: a box
 * on y can be pulled onto x when the player can stand on the square behind y.
 * A square is live when some goal's table reaches it.
 */
function pushDistances(level) {
  const { w, wall, goals, size } = level;
  const tables = [];
  for (const g of goals) {
    const dist = new Int32Array(size).fill(INF);
    dist[g] = 0;
    const queue = [g];
    for (let qi = 0; qi < queue.length; qi++) {
      const x = queue[qi];
      const nd = dist[x] + 1;
      for (let k = 0; k < 4; k++) {
        const d = DY[k] * w + DX[k];
        const y = x - d;
        const behind = y - d;
        if (y < 0 || y >= size || behind < 0 || behind >= size) continue;
        if (wall[y] || wall[behind] || dist[y] !== INF) continue;
        dist[y] = nd;
        queue.push(y);
      }
    }
    tables.push(dist);
  }
  return tables;
}

/** The squares from which some goal is reachable by pushes alone. */
function liveSquares(tables, size) {
  const live = new Uint8Array(size);
  for (const dist of tables) {
    for (let s = 0; s < size; s++) if (dist[s] !== INF) live[s] = 1;
  }
  return live;
}

// Every ordering of n goal indexes, for the exact assignment heuristic.
const PERMUTATIONS = (() => {
  const out = {};
  for (const n of [1, 2, 3, 4, 5]) {
    const base = Array.from({ length: n }, (_, i) => i);
    const list = [];
    const permute = (a, i) => {
      if (i === a.length) {
        list.push(a.slice());
        return;
      }
      for (let j = i; j < a.length; j++) {
        [a[i], a[j]] = [a[j], a[i]];
        permute(a, i + 1);
        [a[i], a[j]] = [a[j], a[i]];
      }
    };
    permute(base, 0);
    out[n] = list;
  }
  return out;
})();

/**
 * The heuristic: the cheapest way to match every box to a distinct goal under
 * the push-distance tables. Exact (every permutation) for up to 5 boxes, a
 * greedy match above that. Exact matching is admissible and consistent: one
 * push moves one box by one push-step, so the value drops by at most 1.
 */
function heuristic(boxes, tables) {
  const n = boxes.length;
  if (n === 0) return 0;
  if (n <= 5) {
    let best = INF;
    for (const perm of PERMUTATIONS[n]) {
      let sum = 0;
      for (let i = 0; i < n; i++) {
        const c = tables[perm[i]][boxes[i]];
        if (c === INF) {
          sum = INF;
          break;
        }
        sum += c;
      }
      if (sum < best) best = sum;
    }
    return best;
  }
  // Greedy match: each box in turn takes its cheapest free goal.
  const taken = new Uint8Array(tables.length);
  let sum = 0;
  for (const b of boxes) {
    let best = INF;
    let bestJ = -1;
    for (let j = 0; j < tables.length; j++) {
      if (taken[j]) continue;
      const c = tables[j][b];
      if (c < best) {
        best = c;
        bestJ = j;
      }
    }
    if (bestJ < 0 || best === INF) return INF;
    taken[bestJ] = 1;
    sum += best;
  }
  return sum;
}

/**
 * The squares the player reaches from `from` with `occupied` boxes in the
 * way, marked in `seen` with `stamp`; returns the region's lowest square,
 * which names the region in a state's key.
 */
function floodRegion(level, occupied, from, seen, stamp, stack) {
  const { w, wall } = level;
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n];
    if (p < low) low = p;
    for (let k = 0; k < 4; k++) {
      const q = p + DY[k] * w + DX[k];
      if (wall[q] === 0 && occupied[q] === 0 && seen[q] !== stamp) {
        seen[q] = stamp;
        stack[n++] = q;
      }
    }
  }
  return low;
}

/** Whether the box just pushed to `t` sits in a frozen 2x2 block. */
function frozenBlock(level, occupied, goal, t) {
  const { w, wall } = level;
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (!(wall[a] || occupied[a])) continue;
    if (!(wall[b] || occupied[b])) continue;
    if (!(wall[c] || occupied[c])) continue;
    if (!(wall[d] || occupied[d])) continue;
    if ((occupied[a] && !goal[a]) || (occupied[b] && !goal[b]) || (occupied[c] && !goal[c]) || (occupied[d] && !goal[d])) {
      return true;
    }
  }
  return false;
}

/** A binary heap ordered by priority, then deeper first, then first found. */
class Heap {
  constructor() {
    this.a = [];
  }
  get size() {
    return this.a.length;
  }
  push(e) {
    const a = this.a;
    let i = a.length;
    a.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!Heap.less(a[i], a[p])) break;
      [a[i], a[p]] = [a[p], a[i]];
      i = p;
    }
  }
  pop() {
    const a = this.a;
    const top = a[0];
    const last = a.pop();
    if (a.length > 0) {
      a[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= a.length) break;
        if (c + 1 < a.length && Heap.less(a[c + 1], a[c])) c++;
        if (!Heap.less(a[c], a[i])) break;
        [a[i], a[c]] = [a[c], a[i]];
        i = c;
      }
    }
    return top;
  }
  static less(x, y) {
    return x[0] < y[0] || (x[0] === y[0] && (x[1] > y[1] || (x[1] === y[1] && x[2] < y[2])));
  }
}

/**
 * Push search over one level (a room, or a whole level as a fallback).
 * A* when greedy is false (priority g + h, first solved state popped is
 * push-optimal), best-first on h alone when true. Returns {pushes, moves}
 * or null when the budget runs out or no solution exists. `budget` is the
 * run-wide expansion account {used, limit}; the search stops when it is spent.
 */
function searchPushes(level, cap, greedy, budget) {
  const { w, wall, goal, boxes: startBoxes, player: startPlayer, size } = level;
  const n = startBoxes.length;
  const solved = (boxes) => boxes.every((b) => goal[b]);
  if (solved(startBoxes)) return { pushes: 0, moves: '' };
  const tables = pushDistances(level);
  const live = liveSquares(tables, size);
  if (startBoxes.some((b) => !live[b])) return null;

  const numericKey = n <= 5;
  const keyOf = (boxes, low) => {
    if (!numericKey) return `${low}:${boxes.join(',')}`;
    let k = low;
    for (const b of boxes) k = k * size + b;
    return k;
  };

  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;

  // The states in the order found: boxes, the player's square after the push
  // that made them, the parent, and the push itself (from square, to square).
  const sBoxes = [startBoxes.slice()];
  const sPlayer = [startPlayer];
  const sParent = [-1];
  const sFrom = [-1];
  const sTo = [-1];
  const sKey = [];
  const best = new Map();
  for (const b of startBoxes) occupied[b] = 1;
  const low0 = floodRegion(level, occupied, startPlayer, seen, ++stamp, stack);
  for (const b of startBoxes) occupied[b] = 0;
  sKey.push(keyOf(startBoxes, low0));
  best.set(sKey[0], 0);

  const heap = new Heap();
  const h0 = heuristic(startBoxes, tables);
  if (h0 === INF) return null;
  heap.push([h0, 0, 0, 0]);
  let seq = 1;
  let expansions = 0;

  while (heap.size > 0) {
    const [, g, , s] = heap.pop();
    if (best.get(sKey[s]) !== s) continue; // a better route to this state exists
    const boxes = sBoxes[s];
    if (solved(boxes)) return { pushes: g, moves: reconstruct(level, sBoxes, sParent, sFrom, sTo, s) };
    if (expansions >= cap || budget.used >= budget.limit) return null;
    expansions++;
    budget.used++;
    for (const b of boxes) occupied[b] = 1;
    floodRegion(level, occupied, sPlayer[s], seen, ++stamp, stack);
    const here = stamp;
    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi];
      for (let k = 0; k < 4; k++) {
        const d = DY[k] * w + DX[k];
        const t = b + d;
        if (seen[b - d] !== here || wall[t] || occupied[t] || !live[t]) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        const frozen = frozenBlock(level, occupied, goal, t);
        let key = null;
        let next = null;
        let h = INF;
        if (!frozen) {
          next = boxes.slice();
          next[bi] = t;
          next.sort((x, y) => x - y);
          h = heuristic(next, tables);
          if (h !== INF) key = keyOf(next, floodRegion(level, occupied, b, mark, ++stamp, stack));
        }
        occupied[t] = 0;
        occupied[b] = 1;
        if (frozen || h === INF || key === null) continue;
        const prev = best.get(key);
        if (prev !== undefined && g + 1 >= gOf(prev)) continue;
        const t2 = sBoxes.length;
        sBoxes.push(next);
        sPlayer.push(b);
        sParent.push(s);
        sFrom.push(b);
        sTo.push(t);
        sKey.push(key);
        best.set(key, t2);
        const ng = g + 1;
        heap.push([greedy ? h : ng + h, ng, seq++, t2]);
      }
    }
    for (const b of boxes) occupied[b] = 0;
  }
  return null;

  // The pushes a stored state took: its depth in the parent chain. Stored
  // states are few enough that walking the chain is cheap, and only runs
  // when a key repeats.
  function gOf(state) {
    let depth = 0;
    for (let s = state; sParent[s] >= 0; s = sParent[s]) depth++;
    return depth;
  }
}

/** The moves of a found path: walk to each push's square, then push. */
function reconstruct(level, sBoxes, sParent, sFrom, sTo, found) {
  const path = [];
  for (let s = found; s > 0; s = sParent[s]) path.push(s);
  path.reverse();
  const { w, wall, size } = level;
  const occupied = new Uint8Array(size);
  let moves = '';
  let player = level.player;
  let boxes = level.boxes;
  for (const s of path) {
    const from = sFrom[s];
    const to = sTo[s];
    const d = to - from;
    const k = d === -w ? 0 : d === w ? 1 : d === -1 ? 2 : 3;
    for (const b of boxes) occupied[b] = 1;
    moves += walkTo(level, occupied, player, from - d);
    for (const b of boxes) occupied[b] = 0;
    moves += PUSH[k];
    player = from;
    boxes = sBoxes[s];
  }
  return moves;
}

/** The shortest walk from `from` to `to` around walls and boxes, in LURD. */
function walkTo(level, occupied, from, to) {
  if (from === to) return '';
  const { w, wall, size } = level;
  const prev = new Int32Array(size).fill(-1);
  const how = new Uint8Array(size);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi];
    if (p === to) break;
    for (let k = 0; k < 4; k++) {
      const q = p + DY[k] * w + DX[k];
      if (q < 0 || q >= size || wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = k;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkTo: the push square is out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]) s = STEP[how[p]] + s;
  return s;
}

/** The square the player stands on after `moves` (LURD, either case). */
function endOf(w, from, moves) {
  let p = from;
  for (const ch of moves) {
    const c = ch.toLowerCase();
    p += c === 'u' ? -w : c === 'd' ? w : c === 'l' ? -1 : 1;
  }
  return p;
}

/**
 * The level's rooms: cut the dead squares of the player's row (the hallway:
 * a box there reaches no goal) out of the floor, and each connected part left
 * that holds a box is a room. Null when some part holds more or fewer boxes
 * than goals, or a box starts in the hallway: then the level is no set of
 * rooms, and the caller falls back to a whole-level search.
 */
function splitRooms(level) {
  const { w, wall, goals, boxes, player, size } = level;
  const tables = pushDistances(level);
  const live = liveSquares(tables, size);
  const hallRow = Math.floor(player / w);
  const cut = new Uint8Array(size);
  for (let x = 0; x < w; x++) {
    const i = hallRow * w + x;
    if (!wall[i] && !live[i]) cut[i] = 1;
  }
  if (boxes.some((b) => cut[b])) return null;
  const comp = new Int32Array(size).fill(-1);
  const parts = [];
  for (let s = 0; s < size; s++) {
    if (wall[s] || cut[s] || comp[s] >= 0) continue;
    const id = parts.length;
    const squares = [s];
    comp[s] = id;
    for (let i = 0; i < squares.length; i++) {
      const p = squares[i];
      for (let k = 0; k < 4; k++) {
        const q = p + DY[k] * w + DX[k];
        if (q < 0 || q >= size || wall[q] || cut[q] || comp[q] >= 0) continue;
        comp[q] = id;
        squares.push(q);
      }
    }
    parts.push({ boxes: [], goals: [] });
  }
  for (const b of boxes) parts[comp[b]].boxes.push(b);
  for (const g of goals) {
    if (comp[g] < 0) return null;
    parts[comp[g]].goals.push(g);
  }
  if (parts.some((c) => c.boxes.length !== c.goals.length)) return null;
  return parts.filter((c) => c.boxes.length > 0);
}

/** Solve one level's text: the moves, or '' when no solution was found. */
function solveLevel(text, budget) {
  const level = parseLevel(text);
  if (level.player < 0) return '';
  if (level.boxes.every((b) => level.goal[b])) return '';
  const rooms = splitRooms(level);
  if (rooms !== null) {
    let player = level.player;
    let moves = '';
    let ok = true;
    for (const room of rooms) {
      // The room's own board: the level's walls, with the other rooms' boxes
      // standing as walls (on their goals once their room is done).
      const wall = level.wall.slice();
      for (const other of rooms) {
        if (other === room) continue;
        for (const b of other.done ? other.goals : other.boxes) wall[b] = 1;
      }
      const goal = new Uint8Array(level.size);
      for (const g of room.goals) goal[g] = 1;
      const sub = {
        w: level.w,
        h: level.h,
        size: level.size,
        wall,
        goal,
        goals: room.goals.slice().sort((a, b) => a - b),
        boxes: room.boxes.slice().sort((a, b) => a - b),
        player,
      };
      let res = searchPushes(sub, OPTIMAL_CAP, false, budget);
      if (res === null) res = searchPushes(sub, GREEDY_CAP, true, budget);
      if (res === null) {
        ok = false;
        break;
      }
      moves += res.moves;
      player = endOf(level.w, player, res.moves);
      room.done = true;
    }
    if (ok) return moves;
  }
  // No rooms, or a room beat both searches: one greedy run over the whole level.
  const whole = searchPushes(level, WHOLE_CAP, true, budget);
  return whole === null ? '' : whole.moves;
}

const budget = { used: 0, limit: GLOBAL_CAP };
for (const id of readdirSync(INPUT).sort()) {
  let moves = '';
  try {
    moves = solveLevel(readFileSync(join(INPUT, id, 'level.txt'), 'utf8'), budget);
  } catch {
    moves = '';
  }
  mkdirSync(join(OUTPUT, id), { recursive: true });
  writeFileSync(join(OUTPUT, id, 'solution.txt'), `${moves}\n`);
}
