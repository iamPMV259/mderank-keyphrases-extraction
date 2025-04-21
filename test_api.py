from requests.models import Response


import requests
import time

doc = """Revealed: America’s Crypto Clampdown on Hamas
Back in April 2023, Hamas declared it was suspending all fundraising efforts through Bitcoin — primarily because donors were facing “hostile” activity.
But a new statement from the U.S. Justice Department suggests this pause didn’t last long.
It’s been revealed that American investigators have seized more than $200,000 worth of crypto intended for the militant group, which has been designated as a terrorist organization by many Western nations.
Further scrutiny of the addresses in question reveals that over $1.5 million in digital assets had been laundered since October of last year.
American officials have vowed to “search high and low for every cent of money going to fund Hamas” — with U.S. Attorney Edward R. Martin Jr. adding:
“Hamas is responsible for the death of many U.S. and Israeli nationals, and we will stop at nothing to stop their campaign of terror and murder.”
FBI agents added that every seizure of this kind “weakens their ability to function” — and deprives them of “resources they need to continue their illicit activity.”
According to this statement, the crypto donations had been solicited through a group chat on an encrypted messaging platform. Hamas supporters were asked to send contributions to a rotating set of 17 addresses, with funds then laundered through a slew of exchanges.
Blockchain analytics was used to bring these shady transactions to light — with Chainalysis revealing that both Binance and Tether cooperated with law enforcement by freezing funds so they could be seized by the Justice Department. In a statement provided to Chainalysis, Tether’s chief executive Paolo Ardoino said:
“Tether is proud to proactively support global law enforcement efforts in combating the illicit use of stablecoins. The notion that criminals can use stablecoins to hide is simply false — every asset is on-chain, visible, and traceable in real time.”
Ardoino vowed that Tether “will continue to work proactively with authorities around the world to safeguard the integrity of the financial ecosystem.”
Chainalysis went on to add that it believes this operation will serve as an effective deterrent for terrorist organizations in the future.
Crypto ‘Weak’ as a Terror Fundraising Tool
Back in 2023 — just a couple of weeks after the October 7th attack — the blockchain analytics firm Elliptic stressed that the amounts terror groups are raising through cryptocurrency “remain tiny compared to other funding sources.”
While Hamas was the first terrorist organization to start soliciting Bitcoin donations in 2019, both Israel and the U.S. have been engaged in aggressive efforts to seize funds — collaborating with exchanges to deactivate their accounts. Elliptic added:
“This illustrates the weakness of crypto as a terrorism fundraising tool. The transparency of the blockchain allows illicit funds to be traced, and in some cases linked to real-world identities.”
At the time, Elliptic was responding to a Wall Street Journal report that had vastly overestimated the amount of crypto that Hamas had received — with a letter written by lawmakers inaccurately stating that more than $130 million had been raised by Hamas and Palestinian Islamic Jihad from August 2021 to June 2023. It went on to warn:
“Careful and detailed understanding of blockchain analysis is needed whenever approaching a nuanced and sensitive topic such as this, and the full context of any analysis should be provided by those using these insights.”
Clamping Down on Hamas
According to the Congressional Research Service, U.S. authorities have adopted a multi-pronged approach to make it prohibitively difficult for Hamas-linked entities to solicit crypto donations. This has included collaborating with other jurisdictions on sanctions, alerting financial institutions about this threat, and fining the trading platforms that have facilitated these transactions.
Binance’s $4.4 billion fine for violations of the Bank Secrecy Act was directly related to how the exchange had failed to disclose suspicious transfers made by the likes of Hamas and North Korea . That also led to the shock resignation of Changpeng Zhao , with then treasury secretary Janet Yellen declaring:
“Binance turned a blind eye to its legal obligations in the pursuit of profit. Its willful failures allowed money to flow to terrorists, cybercriminals, and child abusers through its platform.”
Despite the traceability of cryptocurrencies overall, malicious actors have found ways to launder digital assets, as seen following the Lazarus Group’s audacious hack of the Bybit exchange.
Going forward, law enforcement agencies and crypto firms alike will need to remain vigilant.
"""

start_time: float = time.time()

res: Response = requests.post("http://localhost:5040/mderank", headers={"accept": "application/json", "Content-Type": "application/json"}, json={
    "text": doc,
    "top_k": 15
})

print(res.text)
print("Running time = ", time.time() - start_time)