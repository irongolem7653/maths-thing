import math
import numpy as np
import pygame

STAGE_W, STAGE_H = 480, 360      
SCALE = 2                        
FOV = 360                        
Z_SHIFT = 4                      
START_EQUATION = "x*x+y*y+z*z=2"
START_ZOOM = 1.5                 
START_RES = 25                   
MIN_RES, MAX_RES = 7, 50
START_CX = -25                   
BACKGROUND = (66, 66, 66)
TRIG_DEGREES = True


def _trig(fn):
    return (lambda v: fn(np.radians(v))) if TRIG_DEGREES else fn


def _inverse_trig(fn):
    return (lambda v: np.degrees(fn(v))) if TRIG_DEGREES else fn


FUNCS = {
    "sin": _trig(np.sin), "cos": _trig(np.cos), "tan": _trig(np.tan),
    "asin": _inverse_trig(np.arcsin), "acos": _inverse_trig(np.arccos),
    "atan": _inverse_trig(np.arctan),
    "sqrt": np.sqrt, "abs": np.abs, "floor": np.floor, "ceil": np.ceil,
    "round": lambda v: np.floor(v + 0.5),     
    "ln": np.log, "log": np.log10,             
    "exp": np.exp,
}
CONSTANTS = {"pi": math.pi, "e": math.e}
NAMES = sorted(list(FUNCS) + list(CONSTANTS), key=len, reverse=True)


def tokenize(text):
    s = text.lower().replace(" ", "").replace("**", "^")
    tokens = []
    i = 0
    while i < len(s):
        c = s[i]
        if c.isdigit() or c == ".":
            j = i
            while j < len(s) and (s[j].isdigit() or s[j] == "."):
                j += 1
            try:
                tokens.append(("num", float(s[i:j])))
            except ValueError:
                raise ValueError(f"bad number '{s[i:j]}'")
            i = j
        elif c.isalpha():
            for name in NAMES:
                if s.startswith(name, i):
                    tokens.append(("func" if name in FUNCS else "const", name))
                    i += len(name)
                    break
            else:
                if c in "xyz":
                    tokens.append(("var", c))
                    i += 1
                else:
                    raise ValueError(f"unknown letter '{c}' (variables are x, y, z)")
        elif c in "+-*/%^()=":
            tokens.append(("op", c))
            i += 1
        else:
            raise ValueError(f"unexpected character '{c}'")
    return tokens


class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.i = 0

    def peek(self):
        return self.tokens[self.i] if self.i < len(self.tokens) else (None, None)

    def take(self):
        tok = self.peek()
        self.i += 1
        return tok

    def parse(self):
        left = self.expr()
        if self.peek() == ("op", "="):
            self.take()
            right = self.expr()
            node = ("-", left, right)          
        else:
            node = left                        
        if self.i < len(self.tokens):
            raise ValueError(f"unexpected '{self.peek()[1]}'")
        return node

    def expr(self):
        node = self.term()
        while self.peek() in (("op", "+"), ("op", "-")):
            op = self.take()[1]
            node = (op, node, self.term())
        return node

    def term(self):
        node = self.unary()
        while True:
            kind, val = self.peek()
            if (kind, val) in (("op", "*"), ("op", "/"), ("op", "%")):
                self.take()
                node = (val, node, self.unary())
            elif kind in ("num", "var", "const", "func") or (kind, val) == ("op", "("):
                node = ("*", node, self.unary())      
            else:
                return node

    def unary(self):
        if self.peek() in (("op", "-"), ("op", "+")):
            op = self.take()[1]
            operand = self.unary()
            return ("neg", operand) if op == "-" else operand
        return self.power()

    def power(self):
        base = self.atom()
        if self.peek() == ("op", "^"):
            self.take()
            return ("^", base, self.unary())          
        return base

    def atom(self):
        kind, val = self.take()
        if kind == "num":
            return ("num", val)
        if kind == "var":
            return ("var", val)
        if kind == "const":
            return ("num", CONSTANTS[val])
        if kind == "func":
            nxt = self.peek()
            arg = self.unary() if nxt in (("op", "-"), ("op", "+")) else self.atom()
            return ("func", val, arg)
        if (kind, val) == ("op", "("):
            node = self.expr()
            if self.take() != ("op", ")"):
                raise ValueError("missing )")
            return node
        raise ValueError("equation ended too early" if kind is None else f"unexpected '{val}'")


def power(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    magnitude = np.exp(np.log(np.abs(a)) * b)
    sign = np.where((a < 0) & (np.floor(b) % 2 == 1), -1.0, 1.0)
    out = sign * magnitude
    return np.where(a == 0, np.where(b == 0, 1.0, np.where(b > 0, 0.0, np.inf)), out)


def evaluate(node, env):
    kind = node[0]
    if kind == "num":
        return node[1]
    if kind == "var":
        return env[node[1]]
    if kind == "neg":
        return -evaluate(node[1], env)
    if kind == "func":
        return FUNCS[node[1]](evaluate(node[2], env))
    a = evaluate(node[1], env)
    b = evaluate(node[2], env)
    if kind == "+":
        return a + b
    if kind == "-":
        return a - b
    if kind == "*":
        return a * b
    if kind == "/":
        return a / b
    if kind == "%":
        return np.mod(a, b)
    if kind == "^":
        return power(a, b)
    raise ValueError(f"bad node {kind}")


def compile_equation(text):
    tree = Parser(tokenize(text)).parse()

    def f(x, y, z):
        with np.errstate(all="ignore"):                  
            result = evaluate(tree, {"x": x, "y": y, "z": z})
        return np.broadcast_to(np.asarray(result, dtype=float), x.shape)

    return f


CORNERS = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
EDGES = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (3, 7), (2, 6)]
_TRI_ROWS = (
    'fffffffffffff', '038ffffffffff', '091ffffffffff', '381189fffffff',
    '2b3ffffffffff', '80bb02fffffff', '32b109fffffff', 'b12b91b89ffff',
    '1a2ffffffffff', '03821afffffff', 'a29920fffffff', '8238a289affff',
    'b3aa31fffffff', 'a01a80ab8ffff', '9309b39abffff', '89bb9afffffff',
    '487ffffffffff', '743340fffffff', '487091fffffff', '149174137ffff',
    '874b32fffffff', '4b742b402ffff', '091874b32ffff', '74bb42249291f',
    '48721afffffff', '743340a21ffff', 'a29920748ffff', 'a23a343749a4f',
    '1a33ab487ffff', 'ab1b741b4140f', '7489309b39abf', '74b49b9abffff',
    '945ffffffffff', '945803fffffff', '450051fffffff', '584538513ffff',
    '945b32fffffff', '2b00b8594ffff', '450051b32ffff', '51412b41b4b8f',
    '1a2594fffffff', '94503821affff', '25a245204ffff', 'a25524423438f',
    'b3aa31459ffff', '459a01a80ab8f', 'b30b05045ab5f', '4585a8ab8ffff',
    '879975fffffff', '390359375ffff', '708710751ffff', '753351fffffff',
    '5977982b3ffff', '2b7279759029f', '2b3708710751f', '2b1b71751ffff',
    '87997521affff', 'a21390359375f', '7585a2852820f', 'a25235375ffff',
    '875859b3a31af', '5b7ab5190ffff', 'b5a75b830ffff', '5b7ab5fffffff',
    '67bffffffffff', '7b6380fffffff', '67b091fffffff', '91881367bffff',
    '327726fffffff', '078067026ffff', '672273910ffff', '678681891261f',
    'b67a21fffffff', '380b67a21ffff', '09229a7b6ffff', '67b8238a289af',
    '7a671a731ffff', '80770660161af', '73630963969af', '67a78a89affff',
    'b68864fffffff', '63b603640ffff', 'b68864109ffff', '1393b6936964f',
    '283248264ffff', '406602fffffff', '910283248264f', '914124264ffff',
    '48668b1a2ffff', '1a263b603640f', 'b64b48a29209f', 'a4964ab23ffff',
    '48343a31a64af', '1a0a60640ffff', '4a69a4083ffff', '4a69a4fffffff',
    '67b459fffffff', '4597b6380ffff', '105504b67ffff', 'b67584538513f',
    '327726945ffff', '594078067026f', '326367105045f', '612516478ffff',
    'a2167b459ffff', '038459b67a21f', '7b625a245204f', '8475a63b2ffff',
    '9457a671a731f', 'a65784190ffff', '43073465affff', 'a65847fffffff',
    '9659b698bffff', 'b63360065059f', 'b65b505108b0f', 'b63653513ffff',
    '985832582526f', '596906026ffff', '165261308ffff', '165261fffffff',
    '21a9659b698bf', '9013b25a6ffff', 'b0820ba65ffff', '3b25a6fffffff',
    '1839815a6ffff', '65a019fffffff', '8305a6fffffff', '65affffffffff',
    'a56ffffffffff', '0386a5fffffff', 'a56910fffffff', '3811896a5ffff',
    '2b36a5fffffff', '80bb0256affff', '1092b36a5ffff', '56ab12b91b89f',
    '561162fffffff', '561162803ffff', '695609620ffff', '625238528589f',
    '36b356315ffff', '801816156b86f', 'b36635530509f', '5696b9b89ffff',
    '56a748fffffff', '034437a56ffff', '56a487091ffff', '6a5149174137f',
    '7486a52b3ffff', 'a564b742b402f', '4876a532b109f', '12ab76954ffff',
    '216615874ffff', '037074216156f', '874695609620f', '723627549ffff',
    '48736b356315f', '50140576bffff', '9546b7083ffff', 'b76954fffffff',
    '6a44a9fffffff', '6a44a9380ffff', '0a106a046ffff', '6a1618138468f',
    '94aa4632bffff', '2b82806a4a94f', 'b320a106a046f', '684b862a1ffff',
    '419421462ffff', '380419421462f', '624420fffffff', '382842462ffff',
    '4696b3963931f', '86b468901ffff', 'b36306046ffff', '86b468fffffff',
    'a76a87a98ffff', '37076a07a0a9f', '6a77a88a1810f', '6a7a17137ffff',
    '32ba76a87a98f', '290a926b7ffff', '08376b12affff', '76b12afffffff',
    '219297987627f', '276372019ffff', '870760620ffff', '723627fffffff',
    '819318b76ffff', 'b76190fffffff', '6b7083fffffff', 'b76ffffffffff',
    '7b55bafffffff', 'a5bb57038ffff', '7b55ba091ffff', '7ba7a5381891f',
    '52a532573ffff', '57a780a70a02f', '09152a532573f', '978579a12ffff',
    '1b217b157ffff', '8031b217b157f', '7b2729209579f', '7958973b2ffff',
    '317715fffffff', '807017157ffff', '093953573ffff', '978579fffffff',
    '8548a58baffff', '03b0b5ba5405f', '1098548a58baf', 'a3b13a954ffff',
    '32882442a4a5f', 'a52542402ffff', '549830a12ffff', '2a1495fffffff',
    '8b4b214b1415f', '05415023bffff', '0b28b0495ffff', '54923bfffffff',
    '485835315ffff', '054150fffffff', '549308fffffff', '549ffffffffff',
    'b47b94ba9ffff', '038b47b94ba9f', 'ba7a107a0704f', '3a1ba3784ffff',
    '32a3a4a94734f', '92a029847ffff', '34704312affff', '784a12fffffff',
    '7b44b99b2921f', '19047823bffff', '7b4b24204ffff', '47823bfffffff',
    '941471731ffff', '784190fffffff', '347043fffffff', '784ffffffffff',
    'ba88a9fffffff', '0393b9ba9ffff', '10a08a8baffff', 'a3b13afffffff',
    '3282a8a98ffff', '92a029fffffff', '830a12fffffff', '2a1ffffffffff',
    '21b19b98bffff', 'b23901fffffff', 'b0820bfffffff', '3b2ffffffffff',
    '183981fffffff', '190ffffffffff', '830ffffffffff', 'fffffffffffff',
)
TRIANGLES = []
for _row in _TRI_ROWS:
    _edges = [int(ch, 16) for ch in _row.rstrip("f")]
    TRIANGLES.append([tuple(_edges[i:i + 3]) for i in range(0, len(_edges), 3)])
assert len(TRIANGLES) == 256


def build_mesh(f, res, zoom):
    n = res + 1
    axis = np.linspace(-zoom, zoom, n)
    Z, Y, X = np.meshgrid(axis, axis, axis, indexing="ij")     
    values = f(X, Y, Z)

    bad = ~np.isfinite(values)
    inside = values < 0

    pattern = np.zeros((res, res, res), dtype=np.int32)        
    touches_bad = np.zeros((res, res, res), dtype=bool)
    for k, (dx, dy, dz) in enumerate(CORNERS):
        part = (slice(dz, dz + res), slice(dy, dy + res), slice(dx, dx + res))
        pattern |= inside[part].astype(np.int32) << k
        touches_bad |= bad[part]

    surface_cubes = (pattern != 0) & (pattern != 255) & ~touches_bad
    vals = values.tolist()                                     
    triangles = []
    for z, y, x in np.argwhere(surface_cubes).tolist():
        corner_value = [vals[z + dz][y + dy][x + dx] for dx, dy, dz in CORNERS]
        for edge_triple in TRIANGLES[pattern[z, y, x]]:
            triangle = []
            for e in edge_triple:
                a, b = EDGES[e]
                t = corner_value[a] / (corner_value[a] - corner_value[b])   
                ax, ay, az = CORNERS[a]
                bx, by, bz = CORNERS[b]
                triangle.append((x + ax + t * (bx - ax),
                                 y + ay + t * (by - ay),
                                 z + az + t * (bz - az)))
            triangles.append(triangle)

    if not triangles:
        return np.zeros((0, 3, 3))
    return np.array(triangles) / res * 2 - 1


def triangle_normals(tris):
    x1, y1, z1 = tris[:, 0].T
    x2, y2, z2 = tris[:, 1].T
    x3, y3, z3 = tris[:, 2].T
    wx = (y2 - y1) * (z1 - z3) - (z2 - z1) * (y1 - y3)
    wy = (x2 - x1) * (z1 - z3) - (z2 - z1) * (x1 - x3)
    wz = (x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)
    w = np.stack([wx, wy, wz], axis=1)
    length = np.linalg.norm(w, axis=1)
    keep = length > 1e-12                                       
    return w[keep] / length[keep, None], keep


def rotate_and_project(points, cx, cy, cz):
    sx, ccx = math.sin(math.radians(cx)), math.cos(math.radians(cx))
    sy, ccy = math.sin(math.radians(cy)), math.cos(math.radians(cy))
    sz, ccz = math.sin(math.radians(cz)), math.cos(math.radians(cz))
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    x, z = z * sy + x * ccy, z * ccy - x * sy                   
    x, y = x * ccz - y * sz, x * sz + y * ccz                   
    y, z = y * ccx - z * sx, y * sx + z * ccx                   
    depth = z + Z_SHIFT
    return x * FOV / depth, y * FOV / depth, depth              


def light_vector(cx, cy):
    a, b = math.radians(cx), math.radians(cy)
    return np.array([math.sin(a) * math.cos(b), math.cos(a), math.sin(a) * math.sin(b)])


GREEN, RED, YELLOW, BLUE, PURPLE = (134, 251, 96), (251, 96, 96), (251, 222, 96), (96, 124, 251), (203, 96, 251)
BUTTONS = [
    (-235, -162, 15, 15, GREEN, "+", "range_in"),      
    (-215, -162, 15, 15, RED, "-", "range_out"),       
    (-235, -142, 15, 15, YELLOW, "R", "range_reset"),  
    (-215, -142, 15, 15, BLUE, ".", "range_ask"),      
    (-235, 155, 55, 18, PURPLE, "change", "equation_ask"),  
    (-235, -112, 15, 15, GREEN, "+", "res_up"),        
    (-215, -112, 15, 15, RED, "-", "res_down"),        
    (-235, -92, 15, 15, YELLOW, "R", "res_reset"),     
    (-215, -92, 15, 15, BLUE, ".", "res_ask"),         
]
TEXT_COLOUR = (251, 251, 251)


class App:
    def __init__(self):
        pygame.init()
        pygame.key.set_repeat(400, 40)                      
        self.screen = pygame.display.set_mode((STAGE_W * SCALE, STAGE_H * SCALE))
        pygame.display.set_caption("3D equation grapher")
        self.font = pygame.font.Font(None, 13 * SCALE)
        self.clock = pygame.time.Clock()

        self.equation = START_EQUATION
        self.zoom = START_ZOOM
        self.res = START_RES
        self.cx, self.cy, self.cz = START_CX, 0.0, 0.0     

        self.tris = np.zeros((0, 3, 3))                    
        self.normals = np.zeros((0, 3))
        self.dragging = False
        self.drag_sign = 1
        self.last_mouse = (0, 0)
        self.prompt = None                                  
        self.message = ""                                   
        self.busy = False
        self.scene = pygame.Surface(self.screen.get_size())
        self.scene_dirty = True
        self.rebuild()

    
    @staticmethod
    def to_px(x, y):
        return ((x + STAGE_W / 2) * SCALE, (STAGE_H / 2 - y) * SCALE)

    @staticmethod
    def to_stage(px, py):
        return (px / SCALE - STAGE_W / 2, STAGE_H / 2 - py / SCALE)

    
    def rebuild(self, equation=None, zoom=None, res=None):
        equation = self.equation if equation is None else equation
        zoom = self.zoom if zoom is None else zoom
        res = self.res if res is None else res
        self.busy = True
        self.draw()                                         
        try:
            f = compile_equation(equation)
            tris = build_mesh(f, res, zoom)
            normals, keep = triangle_normals(tris) if len(tris) else (np.zeros((0, 3)), np.zeros(0, bool))
            self.tris, self.normals = tris[keep], normals
            self.equation, self.zoom, self.res = equation, zoom, res
            self.message = "" if len(self.tris) else "nothing to draw: the surface is outside this range"
        except ValueError as err:                      
            self.message = f"can't read equation: {err}"
        self.busy = False
        self.scene_dirty = True

    
    def do_action(self, action):
        if action == "range_in":
            self.rebuild(zoom=self.zoom / 1.5)
        elif action == "range_out":
            self.rebuild(zoom=self.zoom * 1.5)
        elif action == "range_reset":
            self.rebuild(zoom=START_ZOOM)
        elif action == "range_ask":
            self.ask("change the graph range to?", self.set_range)
        elif action == "equation_ask":
            self.ask("change the equation to? type nothing to cancel", self.set_equation)
        elif action == "res_up":
            if self.res < MAX_RES:
                self.rebuild(res=self.res + 1)
        elif action == "res_down":
            if self.res > MIN_RES:
                self.rebuild(res=self.res - 1)
        elif action == "res_reset":
            self.rebuild(res=START_RES)
        elif action == "res_ask":
            self.ask("change the graph resolution to?", self.set_res)

    def set_range(self, answer):
        try:
            value = abs(float(answer))
        except ValueError:
            return
        if value > 0:
            self.rebuild(zoom=value)

    def set_equation(self, answer):
        if answer.strip():
            self.rebuild(equation=answer)

    def set_res(self, answer):
        try:
            value = math.floor(float(answer))
        except ValueError:
            return
        self.rebuild(res=max(MIN_RES, min(MAX_RES, value)))

    
    def ask(self, question, on_answer):
        self.prompt = {"question": question, "text": "", "done": on_answer}

    def on_prompt_key(self, ev):
        p = self.prompt
        if ev.key == pygame.K_RETURN:
            self.prompt = None
            p["done"](p["text"])
        elif ev.key == pygame.K_ESCAPE:
            self.prompt = None
        elif ev.key == pygame.K_BACKSPACE:
            p["text"] = p["text"][:-1]
        elif ev.unicode and ev.unicode.isprintable():
            p["text"] += ev.unicode

    
    def button_at(self, px, py):
        x, y = self.to_stage(px, py)
        for x1, y1, w, h, _colour, _label, action in BUTTONS:
            if x1 < x < x1 + w and y1 - h < y < y1:
                return action
        return None

    def on_mouse_down(self, pos):
        action = self.button_at(*pos)
        if action:
            self.do_action(action)
        else:
            self.dragging = True
            self.last_mouse = pos
            s = math.cos(math.radians(self.cx))
            self.drag_sign = 1 if s >= 0 else -1

    def on_mouse_motion(self, pos):
        if not self.dragging:
            return
        dx = (pos[0] - self.last_mouse[0]) / SCALE          
        dy = -(pos[1] - self.last_mouse[1]) / SCALE         
        self.cy += dx * 0.5 * -self.drag_sign
        self.cx += dy * 1
        self.last_mouse = pos
        self.scene_dirty = True

    
    def render_scene(self):
        self.scene.fill(BACKGROUND)

        if len(self.tris):
            sx, sy, depth = rotate_and_project(self.tris, self.cx, self.cy, self.cz)
            px = (sx + STAGE_W / 2) * SCALE
            py = (STAGE_H / 2 - sy) * SCALE
            screen_pts = np.stack([px, py], axis=-1)            

            
            order = np.argsort(-depth.sum(axis=1))

            
            n = self.normals @ light_vector(self.cx, self.cy)
            brightness = np.clip(n * 35 + 65, 0, 100) * 2.55
            grey = brightness.astype(int)[order].tolist()
            polys = screen_pts[order].tolist()
            for poly, g in zip(polys, grey):
                colour = (g, g, g)
                pygame.draw.polygon(self.scene, colour, poly)
                pygame.draw.polygon(self.scene, colour, poly, 1)  

        
        corners = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
        sx, sy, _ = rotate_and_project(corners, self.cx, self.cy, self.cz)
        pts = [self.to_px(a, b) for a, b in zip(sx.tolist(), sy.tolist())]
        overlay = pygame.Surface(self.scene.get_size(), pygame.SRCALPHA)
        for i in range(8):
            for j in range(i + 1, 8):
                if sum(a != b for a, b in zip(corners[i], corners[j])) == 1:   
                    pygame.draw.line(overlay, (255, 255, 255, 64), pts[i], pts[j], 2 * SCALE)
        self.scene.blit(overlay, (0, 0))

    def text(self, string, x, y, colour=TEXT_COLOUR):
        px, py = self.to_px(x, y)
        self.screen.blit(self.font.render(string, True, colour), (int(px), int(py)))

    def draw_ui(self):
        self.text(self.equation, -235, 175)
        if self.message:
            self.text(self.message, -235, 135, RED)
        mouse = pygame.mouse.get_pos()
        hover = self.button_at(*mouse)
        for x1, y1, w, h, colour, label, action in BUTTONS:
            left, top = self.to_px(x1, y1)
            width = (4 if action == hover else 2) * SCALE // 2    
            pygame.draw.rect(self.screen, colour, (int(left), int(top), w * SCALE, h * SCALE), max(1, width))
            self.text(label, x1 + 5, y1 - 1, colour if label == "change" else TEXT_COLOUR)
        self.text(f"range = {self.zoom:g}", -190, -165)
        self.text(f"res = {self.res}", -190, -115)
        if self.busy:
            self.text("Please wait....", -235, 135)
        if self.prompt:
            box = pygame.Rect(0, STAGE_H * SCALE - 50 * SCALE, STAGE_W * SCALE, 50 * SCALE)
            pygame.draw.rect(self.screen, (30, 30, 30), box)
            self.text(self.prompt["question"], -230, -135)
            self.text(self.prompt["text"] + "_", -230, -155, (134, 251, 96))

    def draw(self):
        if self.scene_dirty:
            self.render_scene()
            self.scene_dirty = False
        self.screen.blit(self.scene, (0, 0))
        self.draw_ui()
        pygame.display.flip()

    
    def run(self):
        running = True
        while running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.KEYDOWN and self.prompt:
                    self.on_prompt_key(ev)
                elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    self.on_mouse_down(ev.pos)
                elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                    self.dragging = False
                elif ev.type == pygame.MOUSEMOTION:
                    self.on_mouse_motion(ev.pos)
            self.draw()
            self.clock.tick(60)
        pygame.quit()


if __name__ == "__main__":
    App().run()
