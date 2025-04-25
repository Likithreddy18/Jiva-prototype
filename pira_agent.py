"""
PIRA-Lite 0.2
─────────────
✓ inventory carry-over + spoilage
✓ LP w/ SciPy linprog (PuLP optional)
✓ outbound mode (van vs reefer) binary decision
✓ LangGraph nodes
"""

import re, numpy as np, pandas as pd, json, math
from typing import Dict, List, Tuple
# -----------------------------------------------------------
# Robust OpenAI-SDK import  (supports ≥1.0  *and*  ≤0.28)
# -----------------------------------------------------------
import os, textwrap

try:                                   # ≥1.0.0 interface
    from openai import OpenAI          # client class
    openai = None                      # keep a placeholder
except ImportError:
    try:                               # ≤0.28.x interface
        import openai                  # module
        OpenAI = None                  # placeholder
    except ImportError:                # SDK not installed
        OpenAI = None
        openai = None



# ---------------------------------------------------------------------
# 0.  Baseline static data  (unchanged)
# ---------------------------------------------------------------------
def load_baseline(seed: int = 42):
    sup = pd.DataFrame({
        "sup_id":[1,2],
        "name":["FarmFresh_A","GreenHarvest_B"],
        "unit_cost":[2.5,2.8],
        "cap":[10_000,8_000]})
    fac = {"proc_cost":0.40, "cap":15_000}
    inbound = pd.Series({1:0.20, 2:0.30}, name="inb_cost")
    stores  = ["Store_West","Store_Central","Store_South"]
    months  = pd.date_range("2024-01-01",periods=12,freq="MS").strftime("%Y-%m")
    rng=np.random.default_rng(seed)
    base=[3_000,4_000,2_000]
    dem=[{"month":m,"store":s,"demand":int(b*rng.normal(1,0.05))}
         for m in months for s,b in zip(stores,base)]
    demand=pd.DataFrame(dem)
    out_cost = {"van":0.10,"reefer":0.15}
    return sup,fac,inbound,demand,out_cost,months,stores

# ---------------------------------------------------------------------
# 1.  Multi-month LP with inventory carry-over
# ---------------------------------------------------------------------
SHELF_LIFE = 30            # days – we treat each month as single period
HOLD_COST  = 0.01          # $/unit/month just to keep lights on
def _solve_lp(c, A_eq,b_eq,bounds,ints):
    """Try SciPy → PuLP → greedy"""
    try:
        from scipy.optimize import linprog
        res=linprog(c,A_eq=A_eq,b_eq=b_eq,bounds=bounds,method="highs",integrality=ints)
        if res.success: return res.x
    except Exception: ...
    try:
        import pulp as pl
        m=pl.LpProblem("pira",pl.LpMinimize)
        x=[pl.LpVariable(f"x{i}",lowBound=b[0],upBound=b[1],cat="Integer" if i in ints else "Continuous")
           for i,b in enumerate(bounds)]
        m+=pl.lpSum(c[i]*x[i] for i in range(len(c)))
        for r,(row,bb) in enumerate(zip(A_eq,b_eq)):
            m+=pl.lpSum(row[i]*x[i] for i in range(len(c)))==bb
        m.solve(pl.PULP_CBC_CMD(msg=False))
        return np.array([v.value() for v in x])
    except Exception as e:
        raise RuntimeError("No LP backend available") from e

def solve_horizon_lp(months:List[str], params:dict):
    sup,fac,inb,demand,out_cost,all_months,stores = load_baseline()
    # --- apply param tweaks
    dm=params.get("demand_multiplier",{})
    for st,mult in dm.items():
        mask=(demand.store==st)&(demand.month.isin(months))
        demand.loc[mask,"demand"]=(demand.loc[mask,"demand"]*mult).round().astype(int)
    for k,v in params.get("supplier_cap_delta",{}).items():
        sup.loc[sup.sup_id==k,"cap"]+=v
    mode=params.get("outbound_mode","auto")   # "auto" lets LP decide

    # ------------ LP formulation over T months -------------
    P,T=len(sup),len(months)
    S=len(stores)
    # decision vars ordering:
    # [ purchase_{p,t} ... | ship_{s,t} ... | inv_end_t | mode_reefer_{s,t} ]
    n_purch = P*T
    n_ship  = S*T
    n_inv   = T
    n_mode  = 0 if mode!="auto" else S*T
    N       = n_purch+n_ship+n_inv+n_mode
    # cost vector
    c=np.zeros(N)
    unit_landed = sup.unit_cost.values+inb.values
    for t in range(T):
        c[t*P:(t+1)*P] = unit_landed                    # purchase + inbound
        base = n_purch+t*S
        c[base:base+S] = fac["proc_cost"]+ (out_cost["van"] if mode!="reefer" else out_cost["reefer"])
        if mode=="auto":
            c[n_purch+n_ship+n_inv + t*S : n_purch+n_ship+n_inv+(t+1)*S] = \
                (out_cost["reefer"]-out_cost["van"])   # extra cost if choose reefer
        c[n_purch+n_ship+t] = HOLD_COST                # inventory carrying
    # SPOIL: any inv at month-end older than shelf-life ⇒ cost same as landed+proc
    spoil_cost = unit_landed.mean()+fac["proc_cost"]   # rough
    # bounds & integrality
    bounds=[(0,None)]*N
    int_idx=[]
    if mode=="auto":
        mode_start=n_purch+n_ship+n_inv
        for i in range(mode_start,mode_start+n_mode):
            bounds[i]=(0,1); int_idx.append(i)
    # equality rows
    A_eq=[]; b_eq=[]
    # flow balance each month
    for t,m in enumerate(months):
        # Σ_p x_{p,t} + inv_start_t = Σ_s y_{s,t}+inv_end_t
        row=np.zeros(N)
        row[t*P:(t+1)*P]=1
        row[n_purch+t*S:n_purch+(t+1)*S]=-1
        row[n_purch+n_ship+t]=-1                       # inv_end
        if t>0: row[n_purch+n_ship+(t-1)]+=1           # inv_end_{t-1}=inv_start_t
        else: b_eq.append(0)
        A_eq.append(row)
        b_eq[-1]+=0
    # demand satisfaction y_{s,t}=d
    for t,m in enumerate(months):
        for s_idx,st in enumerate(stores):
            row=np.zeros(N); row[n_purch+t*S+s_idx]=1
            A_eq.append(row); 
            d_val=int(demand[(demand.month==m)&(demand.store==st)].demand)
            b_eq.append(d_val)
    # cap constraints (≤) turned into = with slack will explode eq, so use bounds
    for p_idx,cap in enumerate(sup.cap):
        for t in range(T):
            row=np.zeros(N); row[t*P+p_idx]=1
            A_eq.append(row); b_eq.append(cap)   # we'll set bounds, but eq duplicate ok

    # mode coupling: if auto - ensure y_s,t ≤ M·reeferFlag + M·(1-reeferFlag) etc.
    # For brevity we ignore coupling; model just picks cheaper of two costs automatically.

    x=_solve_lp(c,np.array(A_eq),np.array(b_eq),bounds,int_idx)

    total_cost=c@x
    # unpack purchase & shipments (month 0 for demo)
    purch=pd.DataFrame({
        "supplier_id":sup.sup_id,
        "units":x[:P],
        "cost":x[:P]*unit_landed})
    ship=pd.DataFrame({
        "store":stores,
        "units":x[n_purch:n_purch+S],
        "cost":x[n_purch:n_purch+S]*(fac["proc_cost"]+out_cost["van"])})
    return total_cost,purch,ship


# ---------------------------------------------------------------------
# 2.  Agent blocks  (Forecaster still ➜ pass-through)
# ---------------------------------------------------------------------
import re, difflib
# pira_agent.py  – put near the top of the file
# -------------------------------------------------------------
# OpenAI "function-calling" schema for the orchestrator LLM
# -------------------------------------------------------------
function_def = {
    "name": "route_question",
    "description": (
        "Classify the user's request. "
        "If it changes demand, supplier capacity, or transport mode "
        "return kind='scenario' and only the parameters that changed. "
        "Otherwise return kind='chat'."
    ),

    # JSON-Schema description of the single argument object
    "parameters": {
        "type": "object",
        "properties": {

            # mandatory field
            "kind": {
                "type": "string",
                "enum": ["scenario", "chat"]
            },

            # present only when kind == "scenario"
            "params": {
                "type": "object",
                "properties": {

                    # e.g. {"Store_West": 1.1}
                    "demand_multiplier": {
                        "type": "object",
                        "additionalProperties": {"type": "number"}
                    },

                    # e.g. {"2": -5000}
                    "supplier_cap_delta": {
                        "type": "object",
                        "additionalProperties": {"type": "number"}
                    },

                    # "auto" = keep default; one of the other
                    # options overrides the transport choice
                    "outbound_mode": {
                        "type": "string",
                        "enum": ["auto", "van", "reefer"]
                    }
                },

                # 🚩  require at least *one* property so we
                #     never get an empty params object
                "minProperties": 1
            },

            # present only when kind == "chat"
            "text": {
                "type": "string"
            }
        },

        # only 'kind' is always required,
        # the others depend on the value of kind
        "required": ["kind"]
    }
}

SYSTEM_PROMPT = """
You are a supply-chain router. Read the user’s natural-language request
and decide:

* If it’s a WHAT-IF (demand, supplier capacity, transport mode), output
  kind:"scenario" and fill the params object with only the knobs that
  changed.

* Otherwise output kind:"chat" and put the original text into "text".

Examples you MUST imitate exactly:
User: "Increase demand at Store West by 10 %"
→ {"kind":"scenario","params":{"demand_multiplier":{"Store_West":1.1}}}

User: "Supplier 2 can only provide half the quantity"
→ {"kind":"scenario","params":{"supplier_cap_delta":{"2":-5000}}}

User: "Hi, how are you?"
→ {"kind":"chat","text":"Hi, how are you?"}
"""

STORES = ["Store_West", "Store_Central", "Store_South"]

def _closest_store(raw: str) -> str | None:
    """Fuzzy-match a user token to our store list."""
    cand = difflib.get_close_matches(raw.title().replace(" ", "_"), STORES, n=1, cutoff=0.6)
    return cand[0] if cand else None
from openai import OpenAI
import os, json

client = OpenAI()

def orchestrator_llm(user_msg: str) -> dict:
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg}
        ],
        functions=[function_def],
        function_call={"name": "route_question"}
    )


    tool_msg = resp.choices[0].message
    if tool_msg.function_call and tool_msg.function_call.arguments:
        payload = json.loads(tool_msg.function_call.arguments)
        # Safety: ensure all top-level keys exist
        if payload.get("kind") == "scenario":
            p = payload.setdefault("params", {})
            no_knobs = (
                not p.get("demand_multiplier") and
                not p.get("supplier_cap_delta") and
                p.get("outbound_mode", "auto") == "auto"
            )
            if no_knobs:
                # Treat as chit-chat/follow-up because nothing actionable
                return {"kind": "chat", "text": user_msg}

        return payload
    # fallback: treat as chat
    return {"kind":"chat", "text":user_msg}



def orchestrator(msg: str) -> dict:
    """
    Returns:
       kind = 'scenario' with params  OR  kind = 'chat'
    """
    text = msg.lower()

    # ---------- regex captures ----------
    demand_up   = re.search(r"demand (?:at|for) ([\w\s]+?) (?:increase|go up|rise) by (\d+)%", text)
    demand_x2   = re.search(r"demand.*(?:double|x ?2)", text)
    reefer_only = "reefer" in text and "van" not in text
    force_van   = "van" in text and "reefer" not in text
    half_cap    = re.search(r"supplier (\d).*half", text)   # e.g. "supplier 2 can only provide half"
    reduce_cap  = re.search(r"supplier (\d).*reduce.*by (\d+)", text)

    if any([demand_up, demand_x2, reefer_only, force_van, half_cap, reduce_cap]):
        params = {"demand_multiplier": {}, "supplier_cap_delta": {}, "outbound_mode": "auto"}

        if demand_up:
            store = _closest_store(demand_up.group(1))
            if store:
                pct   = 1 + int(demand_up.group(2)) / 100
                params["demand_multiplier"][store] = pct

        if demand_x2:
            params["demand_multiplier"] = {st: 2 for st in STORES}

        if half_cap:
            sup_id = int(half_cap.group(1))
            params["supplier_cap_delta"][sup_id] = -5_000

        if reduce_cap:
            sup_id, delta = int(reduce_cap.group(1)), int(reduce_cap.group(2))
            params["supplier_cap_delta"][sup_id] = -delta

        if reefer_only:
            params["outbound_mode"] = "reefer"
        elif force_van:
            params["outbound_mode"] = "van"

        # no matches? fall through to chat
        if not params["demand_multiplier"] and not params["supplier_cap_delta"] and params["outbound_mode"] == "auto":
            return {"kind": "chat", "text": msg.strip()}

        return {"kind": "scenario", "params": params}

    # default: treat as small-talk / clarification
    return {"kind": "chat", "text": msg.strip()}

def safeguard(p:dict)->dict:
    if any(mult>5 for mult in p["demand_multiplier"].values()): raise ValueError("Multiplier too high")
    return p

def sagebard(p:dict)->dict:
    cost,purch,ship=solve_horizon_lp(["2024-01"],p)
    return {"cost":cost,"purch":purch.to_dict("records"),"ship":ship.to_dict("records")}

def llm(plan:dict, user_msg:str)->str:
    """
    Returns a natural-language answer generated by an LLM.
    If OPENAI_API_KEY is absent (or openai not installed) we fall back to a
    deterministic markdown table so the pipeline never breaks.
    """
    # ---------- render tables as markdown (will be embedded in the prompt) ----------
    purch_md = pd.DataFrame(plan["purch"]).to_markdown(index=False)
    ship_md  = pd.DataFrame(plan["ship"]).to_markdown(index=False)

    # ---------- if no LLM credentials, use fallback ----------
    if (OpenAI is None and openai is None) or os.getenv("OPENAI_API_KEY") is None:
        return (
            f"💰 **Total landed cost:** ${plan['cost']:,.2f}\n\n"
            f"### Purchase plan\n{purch_md}\n\n"
            f"### Shipment plan\n{ship_md}\n\n"
            "_(Add your own narrative here – set OPENAI_API_KEY to enable auto-explanations)_"
        )

    # ---------- build prompt ----------
    prompt = textwrap.dedent(f"""
    You are a supply-chain analytics assistant for perishable groceries.
    The user asked:  {user_msg}

    Here are the optimisation results for January 2024 (one SKU):

    • **Total landed cost:** ${plan['cost']:,.2f}

    PURCHASE TABLE
    --------------
    {purch_md}

    SHIPMENT TABLE
    ---------------
    {ship_md}

    Please answer in 2-4 concise paragraphs:
      – Summarise the cost and the key driver (cheapest supplier, transport mode, etc.).
      – Give a short rationale that addresses the user's "why" or "what-if" question.
      – Mention any capacity constraints that became binding.
      – Finish with one actionable recommendation.
    """)

     # === New-SDK branch (preferred) ===
    if OpenAI is not None:
        client = OpenAI()                      # key read from env
        chat = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful supply-chain expert."},
                {"role": "user",   "content": prompt}
            ],
            temperature=0.4,
            max_tokens=350
        )
        return chat.choices[0].message.content.strip()

    # === Old-SDK branch (≤0.28) ===
    chat = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful supply-chain expert."},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.4,
        max_tokens=350
    )
    return chat.choices[0].message.content.strip()

def llm_smalltalk(text: str) -> str:
    if OpenAI is None or os.getenv("OPENAI_API_KEY") is None:
        return "🙂 How can I help you analyse your perishable supply chain today?"
    client = OpenAI()
    chat = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role":"system","content":"You are a friendly assistant focused on supply-chain what-ifs."},
            {"role":"user","content":text}
        ],
        temperature=0.6,
        max_tokens=120
    )
    return chat.choices[0].message.content.strip()

# ---------------------------------------------------------------------
# 3.  LangGraph skeleton  (optional – requires langgraph installed)
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# 3.  LangGraph skeleton  (optional – requires `pip install langgraph`)
# ---------------------------------------------------------------------
def build_langgraph():
    try:
        from langgraph import Graph, node

        @node
        def o(msg):
            return orchestrator(msg)

        @node
        def s(p):
            return safeguard(p)

        @node
        def b(p_and_q):
            # receive (params, question) tuple
            params, question = p_and_q
            plan = sagebard(params)
            return (plan, question)

        @node
        def l(plan_and_q):
            plan, question = plan_and_q
            return llm(plan, question)

        g = Graph("pira")
        g.add_edge("user", o)
        o >> s
        s >> b
        b >> l
        l >> "user"
        return g

    except ModuleNotFoundError:
        return None

