"""One-shot script: apply registry corrections + manual subscription status overrides."""
import pandas as pd

df = pd.read_csv('config/fund_registry.csv', encoding='utf-8-sig', dtype=str)
print(f'Before: {len(df)} rows')

# ── PART 1A: Delete 3 rows ────────────────────────────────────────────────────
to_drop = []
to_drop.extend(df[df['display_name'] == 'Sharp Ibovespa Ativo'].index.tolist())
to_drop.extend(df[df['display_name'] == 'Bahia Long Biased CIC'].index.tolist())
to_drop.extend(df[df['display_name'] == 'AF Invest Geraes 30'].index.tolist())
df = df.drop(index=to_drop).reset_index(drop=True)
print(f'After drops: {len(df)} rows')

# ── PART 1B: Real Investor Long Short → Real Investor Multimercado ────────────
mask = df['display_name'] == 'Real Investor Long Short'
df.loc[mask, 'display_name']    = 'Real Investor Multimercado'
df.loc[mask, 'cnpj']            = '28.911.549/0001-57'
df.loc[mask, 'match_status']    = 'matched'
df.loc[mask, 'source_label']    = 'Real Investor FIC FIF Multimercado'
df.loc[mask, 'notes'] = (
    'CVM175 class: REAL INVESTOR FIC DE FUNDO DE INVESTIMENTO FINANCEIRO MULTIMERCADO '
    '(CNPJ 28.911.549/0001-57); formerly tracked as Real Investor Long Short '
    '(CNPJ 60.335.772/0001-06 = Real Investor Long Short Plus FIC FIF Multimercado)'
)
df.loc[mask, 'subscription_status']            = 'unknown'
df.loc[mask, 'subscription_status_source']     = ''
df.loc[mask, 'subscription_status_checked_at'] = ''
df.loc[mask, 'evidence_note']                  = ''
print('Real Investor corrected.')

# ── PART 2: Manual subscription status overrides ─────────────────────────────
MANUAL_SOURCE = 'manual (funds_subscription_status.xlsx)'
MANUAL_TS     = '2026-04-08T00:00:00Z'
CVM_CONFLICT  = (
    '{status} per manual review (funds_subscription_status.xlsx); '
    'note: CVM legacy cad_fi shows SIT=CANCELADA for this CNPJ — '
    'fund likely migrated to a new legal vehicle or continues under a different registration'
)

def _note(status, conflict=False):
    if conflict:
        return CVM_CONFLICT.format(status=status.capitalize())
    if status == 'open':
        return 'Open for subscriptions per manual review (funds_subscription_status.xlsx)'
    if status == 'closed':
        return 'Closed for subscriptions per manual review (funds_subscription_status.xlsx)'
    return 'Status unconfirmed per manual review (funds_subscription_status.xlsx)'

# (status, cvm_conflict)  — None = not in spreadsheet, preserve existing
MANUAL_MAP = {
    'Verde':                                ('closed',  False),
    'Ita\u00fa Artax':                      ('closed',  False),
    'Absolute Vertex II':                   ('open',    False),
    'Novus Macro':                          ('open',    False),
    'Clave Alpha':                          ('open',    False),
    'K\u00ednitro 30':                      ('open',    False),
    'Ace Capital':                          ('open',    False),
    'Legacy Capital':                       ('open',    False),
    'Ventor Hedge':                         ('open',    True),
    'SPX Nimitz':                           ('open',    False),
    'Ita\u00fa Janeiro':                    ('closed',  False),
    'Occam Retorno Absoluto':               ('open',    False),
    'Capstone Macro':                       ('closed',  False),
    'Vinland Macro Plus':                   ('open',    False),
    'G\u00e1vea Macro':                     ('open',    False),
    'Kapitalo K10':                         ('open',    False),
    'XP Macro':                             ('open',    False),
    'Ibiuna Hedge STH':                     ('closed',  False),
    'Kapitalo Zeta':                        ('open',    False),
    'Mar Absoluto':                         ('open',    False),
    'Vista Multiestr\u00e9gia':             None,
    'Vista Multiestrat\u00e9gia':           ('open',    False),
    'Truxt I Macro':                        ('open',    False),
    'Bahia Mara\u00fa':                     ('open',    False),
    'Kapitalo Kappa':                       ('open',    False),
    'Guepardo':                             ('open',    True),
    'SPX Patriot':                          ('open',    False),
    'Charles River A\u00e7\u00f5es':        ('closed',  False),
    'Ita\u00fa Smart A\u00e7\u00f5es Brasil 50': ('open', False),
    'Real Investor A\u00e7\u00f5es':        ('open',    False),
    'Opportunity L\u00f3gica':              ('closed',  False),
    'Kapitalo Tarkus':                      ('open',    False),
    'Opportunity Selection':                ('closed',  False),
    'Zenith Vit\u00f3ria R\u00e9gia':       ('unknown', False),
    'AF Invest Minas':                      ('open',    False),
    'Constellation':                        ('open',    False),
    'Safra Equity Portfolio Special':       ('unknown', False),
    'Squadra Long-Only':                    ('closed',  False),
    'Dynamo Cougar':                        ('closed',  False),
    'Atmos A\u00e7\u00f5es':               ('closed',  False),
    'Sharp Equity Value':                   ('open',    False),
    'Truxt I Long Biased':                  ('open',    False),
    'SPX Falcon':                           ('closed',  False),
    'P\u00e1tria Long Biased':              ('open',    False),
    'Absolute Pace Long Biased':            ('closed',  False),
    'Guepardo Long Biased':                 ('open',    False),
    'Ita\u00fa V\u00e9rtice Fundamenta LATAM': ('unknown', False),
    'Ibiuna Long Biased':                   ('open',    False),
    'Vinci Total Return':                   ('unknown', False),
    'AZ Quest Top Long Biased':             ('open',    False),
    'Occam Long Biased':                    ('open',    False),
    'JGP Equity':                           ('open',    False),
    'Bahia Long Biased':                    ('unknown', False),
    'Genoa Arpa':                           ('open',    False),
    'Ita\u00fa Optimus Long Biased':        None,
    'IP Value Hedge':                       ('open',    False),
    'Polo Long Bias I':                     ('open',    False),
    'Squadra Long-Biased':                  ('closed',  False),
    'Sharp Long Biased':                    ('open',    False),
    'Truxt I Long Short':                   ('open',    False),
    'Polo Norte I':                         ('open',    True),
    'T\u00e1vola Equity Hedge':             ('closed',  False),
    'Real Investor Multimercado':           ('unknown', False),
    'Oceana Equity Hedge':                  ('unknown', False),
    'Bradesco Equity Hedge':                ('open',    False),
    'Sharp Long Short':                     ('closed',  False),
    'XP Investor Equity Hedge':             ('open',    False),
    'SPX Hornet':                           ('closed',  False),
    'Ibiuna Long Short':                    ('open',    False),
    'AZ Quest Total Return':                ('open',    False),
    'Bahia Una Equity Hedge':               ('open',    False),
    'Polo I':                               ('open',    True),
    'Schroder GAIA Contour Tech':           ('open',    False),
    'Icatu Vanguarda Cr\u00e9dito Privado': ('open',    False),
    'AF Invest Geraes':                     ('open',    False),
    'Western Asset Total Credit':           ('unknown', False),
    'Drys Shelter':                         ('open',    False),
    'AZ Quest Luce':                        ('open',    False),
    'ARX Fuji':                             ('open',    False),
    'Cartor Insignia':                      ('open',    False),
    'Compass Credit':                       ('unknown', False),
    'Riza Lotus':                           ('open',    False),
    'Ita\u00fa Active Fix':                 ('open',    False),
    'Occam Liquidez':                       ('open',    False),
    'Tyton Cr\u00e9dito':                   ('open',    False),
    'Real Investor Cr\u00e9dito Estruturado 30': ('unknown', False),
    'Icatu Vanguarda Credit Plus':          ('open',    False),
    'SPX Seahawk':                          ('open',    False),
    'Polo Total Credit':                    ('open',    False),
    'JGP Cr\u00e9dito Advisory':            ('open',    False),
    'JGP Cr\u00e9dito Ecossistema 360':     ('unknown', False),
    'Ang\u00e1 Cr\u00e9dito Estruturado':   ('open',    False),
    'Vinland Cr\u00e9dito Estruturado':     ('open',    False),
    'Root Capital High Yield':              ('open',    False),
    'Real Investor Cr\u00e9dito Estruturado 90': ('unknown', False),
    'Riza Meyenii 180':                     ('open',    False),
    'P\u00e1tria Cr\u00e9dito Estruturado': ('open',    False),
    'M8 Credit Strategy Plus':             ('open',    False),
    'Polo Cr\u00e9dito Estruturado 90':     ('open',    False),
    'JGP Select Premium':                   ('open',    False),
    'Witpar':                               ('open',    False),
    'Tr\u00edgono Power & Yield 30':        ('unknown', False),
    'Asset Bank':                           ('unknown', False),
    'Apuama Yara':                          ('open',    False),
    'MCA CP':                               ('open',    False),
    'Solis Antares Light':                  ('open',    False),
}

applied   = []
preserved = []
not_found = []

for i, row in df.iterrows():
    name = row['display_name']
    if name not in MANUAL_MAP:
        not_found.append(name)
        continue
    entry = MANUAL_MAP[name]
    if entry is None:
        preserved.append(f'{name} (not in spreadsheet)')
        continue
    status, conflict = entry
    df.at[i, 'subscription_status']            = status
    df.at[i, 'subscription_status_source']     = MANUAL_SOURCE
    df.at[i, 'subscription_status_checked_at'] = MANUAL_TS
    df.at[i, 'evidence_note']                  = _note(status, conflict)
    tag = ' [CVM-conflict override]' if conflict else ''
    applied.append(f'{name} -> {status}{tag}')

df.to_csv('config/fund_registry_corrected.csv', index=False, encoding='utf-8-sig')
print(f'Saved to fund_registry_corrected.csv. Final rows: {len(df)}')
print(f'Status applied   : {len(applied)}')
print(f'Preserved (no XL): {len(preserved)}')
print(f'Not in MANUAL_MAP: {len(not_found)} {not_found}')
print()
print('=== APPLIED ===')
for x in applied: print(' ', x)
print()
print('=== PRESERVED / NOT OVERRIDDEN ===')
for x in preserved: print(' ', x)
