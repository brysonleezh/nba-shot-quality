"""
Merge NBA Draft Combine Shooting Drills data (from nba.com/stats/draft/combine-shooting-drills)
into data/combine_2026.json.

New / updated fields per player:
  OFF_DRIB_*     : already existed, now also fills null gaps
  SPOT_UP_*      : maps to existing COLLEGE_CORNER_LEFT_*
  THREE_PT_STAR_*: maps to existing ON_MOVE_COLLEGE_*
  THREE_PT_SIDE_*: NEW
  MIDRANGE_STAR_*: NEW (most players null)
  MIDRANGE_SIDE_*: NEW (most players null)
  FREETHROW_*    : NEW
"""
import json
from pathlib import Path

# ── Raw data pasted from nba.com ──────────────────────────────────────────────
# Columns: name/pos | off_drib FGM FGA | spot_up FGM FGA | 3pt_star FGM FGA
#          | mid_star FGM FGA | 3pt_side FGM FGA | mid_side FGM FGA | ft FGM FGA
RAW = """Hopkins, Bryce	PF	19	30	17	25	12	25	-	-	14	26	-	-	8	10
Mitchell, Dillon	PF	17	30	6	25	9	25	-	-	10	25	-	-	8	10
Mara, Aday	C	13	30	12	25	16	25	-	-	15	24	-	-	7	10
Boswell, Kylan	PG	14	30	12	25	12	25	-	-	13	29	-	-	8	10
Karaban, Alex	PF	21	30	17	25	18	25	-	-	21	25	-	-	8	10
Onyenso, Ugonna	C	17	30	14	25	5	25	-	-	-	-	16	26	9	10
Momcilovic, Milan	PF	22	30	17	25	16	25	-	-	16	28	-	-	10	10
Bradley, Jaden	PG	19	30	10	25	12	25	-	-	12	26	-	-	8	10
Philon, Labaron	PG	23	30	14	25	15	25	-	-	14	26	-	-	9	10
Evans, Isaiah	SG	21	30	14	25	17	25	-	-	15	26	-	-	9	10
Cofie, Jacob	PF	14	30	11	25	14	25	-	-	12	26	-	-	9	10
Carr, Cameron	SG	22	30	14	25	11	25	-	-	13	29	-	-	7	10
Swain, Dailyn	SF	22	30	14	25	12	25	-	-	17	24	-	-	7	10
Thornton, Bruce	PG	20	30	10	25	17	25	-	-	15	27	-	-	7	10
Lendeborg, Yaxel	PF	17	30	16	25	12	25	-	-	14	29	-	-	8	10
Uzan, Milos	PG	24	30	17	25	13	25	-	-	15	27	-	-	9	10
Blackwell, John	SG	18	30	16	25	15	25	-	-	16	26	-	-	7	10
Chinyelu, Rueben	C	18	30	12	25	11	25	-	-	6	24	-	-	6	10
Oweh, Otega	SG	23	30	12	25	12	25	-	-	13	24	-	-	6	10
Johnson, Jr., Morez	PF	15	30	9	25	17	25	-	-	13	29	-	-	9	10
Acuff, Jr., Darius	PG	24	30	19	25	13	25	-	-	13	26	-	-	7	10
Jefferson, Joshua	PF	16	30	11	25	15	25	-	-	15	27	-	-	5	10
Bidunga, Flory	C	8	30	15	25	-	-	13	25	-	-	13	24	10	10
Wagler, Keaton	PG	25	30	11	25	12	25	-	-	12	28	-	-	10	10
Kaufman-Renn, Trey	PF	17	30	11	25	15	25	-	-	14	24	-	-	6	10
Flemings, Kingston	PG	26	30	15	25	19	25	-	-	18	28	-	-	8	10
Bilodeau, Tyler	PF	19	30	19	25	16	25	-	-	17	25	-	-	10	10
Fears, Jr., Jeremy	PG	21	30	13	25	13	25	-	-	15	25	-	-	8	10
Stirtz, Bennett	PG	22	30	20	25	16	25	-	-	21	28	-	-	9	10
Allen, Amari	SF	20	30	13	25	13	25	-	-	17	26	-	-	9	10
Lopez, Karim	PF	16	30	14	25	10	25	-	-	14	28	-	-	10	10
Nickel, Tyler	SF	19	30	16	25	18	25	-	-	18	28	-	-	8	10
Okpara, Felix	C	11	30	7	25	10	25	-	-	11	25	-	-	10	10
Able, Matthew	SG	19	30	17	25	13	25	-	-	14	25	-	-	8	10
Quaintance, Jayden	C	11	30	9	25	6	25	-	-	8	24	-	-	7	10
Boozer, Cameron	PF	18	30	19	25	11	25	-	-	14	25	-	-	9	10
Wilson, Caleb	PF	19	30	11	25	12	25	-	-	14	23	-	-	6	10
Castro, Rafael	C	10	30	6	25	6	25	-	-	4	24	-	-	7	10
Graves, Allen	PF	13	30	14	25	14	25	-	-	12	25	-	-	7	10
Reed Jr., Tarris	C	14	30	7	25	8	25	-	-	13	26	-	-	6	10
Dybantsa, A.J.	SF	23	30	14	25	11	25	-	-	12	25	-	-	10	10
Steinbach, Hannes	C	18	30	16	25	9	25	-	-	12	29	-	-	7	10
Ejiofor, Zuby	C	17	30	13	25	12	25	-	-	16	28	-	-	8	10
Okorie, Ebuka	PG	23	30	19	25	11	25	-	-	11	29	-	-	9	10
Tanner, Tyler	PG	25	30	12	25	18	25	-	-	19	30	-	-	7	10
Peterson, Darryn	SG	21	30	19	25	10	25	-	-	13	27	-	-	6	10
Martinelli, Nick	PF	21	30	15	25	12	25	-	-	12	26	-	-	6	10
Veesaar, Henri	C	18	30	11	25	12	25	-	-	15	28	-	-	10	10
Gillespie, Ja'Kobi	PG	20	30	17	25	17	25	-	-	16	27	-	-	7	10
Peat, Koa	PF	15	30	6	25	7	25	-	-	10	25	-	-	7	10
Brown Jr., Mikel	PG	15	30	18	25	15	25	-	-	19	27	-	-	10	10
Stojakovic, Andrej	SF	20	30	11	25	9	25	-	-	14	29	-	-	7	10
Sharp, Emanuel	SG	19	30	18	25	15	25	-	-	13	25	-	-	10	10
Lawal, Tobi	PF	12	30	13	25	11	25	-	-	17	27	-	-	6	10
Conwell, Ryan	SG	28	30	14	25	19	25	-	-	15	28	-	-	10	10
Cenac Jr., Chris	C	15	30	9	25	16	25	-	-	9	26	-	-	7	10
Brown, Maliq	C	15	30	9	25	13	25	-	-	10	26	-	-	3	10
Burries, Brayden	SG	21	30	14	25	17	25	-	-	13	27	-	-	10	10
Yessoufou, Tounde	SF	16	30	11	25	10	25	-	-	13	27	-	-	8	10
Miller, Baba	PF	13	30	13	25	13	25	-	-	14	28	-	-	6	10
Anderson Jr., Christian	PG	27	30	17	25	17	25	-	-	17	28	-	-	9	10
Smith, Braden	PG	20	30	15	25	13	25	-	-	19	28	-	-	10	10
Brazile, Trevon	PF	15	30	14	25	12	25	-	-	13	28	-	-	8	10
Moreno, Malachi	C	17	30	12	25	6	25	-	-	7	22	-	-	7	10
Ament, Nate	PF	22	30	16	25	13	25	-	-	13	25	-	-	9	10
Awaka, Tobe	C	19	30	13	25	9	25	-	-	16	28	-	-	7	10
Boyd, Nick	PG	20	30	14	25	11	25	-	-	13	28	-	-	7	10
Hall, Keyshawn	SF	23	30	18	25	16	25	-	-	10	25	-	-	9	10
Nelson, Izaiyah	C	16	30	9	25	12	25	-	-	12	25	-	-	7	10
Nkrumah, Aaron	SG	14	30	12	25	10	25	-	-	10	24	-	-	8	10
Richmond III, Billy	SF	19	30	13	25	17	25	-	-	13	26	-	-	8	10
Suder, Peter	SG	15	30	13	25	6	25	-	-	8	27	-	-	6	10
Thomas, Meleek	SG	22	30	14	25	13	25	-	-	9	25	-	-	10	10"""

# ── Name mapping: website "Last, First" → JSON "First Last" ──────────────────
NAME_MAP = {
    "Johnson, Jr., Morez":    "Morez Johnson",
    "Acuff, Jr., Darius":     "Darius Acuff Jr.",
    "Fears, Jr., Jeremy":     "Jeremy Fears Jr.",
    "Anderson Jr., Christian":"Christian Anderson",
    "Brown Jr., Mikel":       "Christopher Brown Jr",  # same player, diff name in JSON
    "Cenac Jr., Chris":       "Christopher Cenac Jr.",
    "Reed Jr., Tarris":       "Tarris Reed Jr.",
    "Dybantsa, A.J.":         "Anicet Dybantsa",
    "Boyd, Nick":             "Nicholas Boyd",
    "Ament, Nate":            "Nathaniel Ament",
    "Martinelli, Nick":       "Nick Martinelli",
    "Boozer, Cameron":        "Cameron Boozer",
    "Boswell, Kylan":         "Kylan Boswell",
}

def website_name_to_json(raw: str) -> str:
    raw = raw.strip()
    if raw in NAME_MAP:
        return NAME_MAP[raw]
    # Generic "Last, First" → "First Last"
    if "," in raw:
        parts = raw.split(",", 1)
        last = parts[0].strip()
        first = parts[1].strip()
        return f"{first} {last}"
    return raw


def parse_val(s: str):
    s = s.strip()
    if s == "-" or s == "":
        return None
    return int(s)


def pct(made, attempt):
    if made is None or attempt is None or attempt == 0:
        return None
    return round(made / attempt, 4)


# ── Parse raw table ───────────────────────────────────────────────────────────
parsed: dict[str, dict] = {}
for line in RAW.strip().splitlines():
    cols = line.split("\t")
    # cols[0]=name, cols[1]=pos, then pairs: off_drib(2,3) spot_up(4,5) 3pt_star(6,7)
    #   mid_star(8,9) 3pt_side(10,11) mid_side(12,13) ft(14,15)
    name_raw = cols[0].strip()
    json_name = website_name_to_json(name_raw)

    od_m, od_a     = parse_val(cols[2]),  parse_val(cols[3])
    su_m, su_a     = parse_val(cols[4]),  parse_val(cols[5])
    star_m, star_a = parse_val(cols[6]),  parse_val(cols[7])
    mstar_m, mstar_a = parse_val(cols[8]),  parse_val(cols[9])
    side_m, side_a = parse_val(cols[10]), parse_val(cols[11])
    mside_m, mside_a = parse_val(cols[12]), parse_val(cols[13])
    ft_m, ft_a     = parse_val(cols[14]), parse_val(cols[15])

    parsed[json_name] = {
        # existing fields (fill gaps / confirm)
        "COLLEGE_CORNER_LEFT_MADE":           su_m,
        "COLLEGE_CORNER_LEFT_ATTEMPT":        su_a,
        "COLLEGE_CORNER_LEFT_PCT":            pct(su_m, su_a),
        "OFF_DRIB_COLLEGE_BREAK_LEFT_MADE":   od_m,
        "OFF_DRIB_COLLEGE_BREAK_LEFT_ATTEMPT":od_a,
        "OFF_DRIB_COLLEGE_BREAK_LEFT_PCT":    pct(od_m, od_a),
        "ON_MOVE_COLLEGE_MADE":               star_m,
        "ON_MOVE_COLLEGE_ATTEMPT":            star_a,
        "ON_MOVE_COLLEGE_PCT":                pct(star_m, star_a),
        # new fields
        "THREE_PT_SIDE_MADE":                 side_m,
        "THREE_PT_SIDE_ATTEMPT":              side_a,
        "THREE_PT_SIDE_PCT":                  pct(side_m, side_a),
        "MIDRANGE_STAR_MADE":                 mstar_m,
        "MIDRANGE_STAR_ATTEMPT":              mstar_a,
        "MIDRANGE_STAR_PCT":                  pct(mstar_m, mstar_a),
        "MIDRANGE_SIDE_MADE":                 mside_m,
        "MIDRANGE_SIDE_ATTEMPT":              mside_a,
        "MIDRANGE_SIDE_PCT":                  pct(mside_m, mside_a),
        "FREETHROW_MADE":                     ft_m,
        "FREETHROW_ATTEMPT":                  ft_a,
        "FREETHROW_PCT":                      pct(ft_m, ft_a),
    }

# ── Load and update combine_2026.json ────────────────────────────────────────
path = Path("data/combine_2026.json")
data = json.loads(path.read_text())

matched, unmatched = 0, []
for p in data:
    name = p["PLAYER_NAME"]
    if name in parsed:
        p.update(parsed[name])
        matched += 1
    else:
        unmatched.append(name)

# Report players in website data not found in JSON
not_in_json = [n for n in parsed if not any(p["PLAYER_NAME"] == n for p in data)]

path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

print(f"Updated {matched} players in combine_2026.json")
print(f"\nNot matched in JSON ({len(not_in_json)}): {not_in_json}")
print(f"\nIn JSON but not in website data ({len(unmatched)}): {unmatched}")
