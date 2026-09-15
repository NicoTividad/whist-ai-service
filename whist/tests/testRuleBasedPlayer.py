import unittest

from whist.main import evaluateRuleVsRandom


# ----- THE RULE-BASED PLAYER BEATS RANDOM -----

class TestRuleBasedBeatsRandom(unittest.TestCase):

    def testRuleSideWinsClearMajority(self):
        # a fixed, seeded batch is deterministic, so this result is stable and not flaky
        result = evaluateRuleVsRandom(100, 1, 1) # 100 games, deck and player seeds both starting at 1
        self.assertGreaterEqual(result["ruleWins"], 60)


if __name__ == "__main__":
    unittest.main()
