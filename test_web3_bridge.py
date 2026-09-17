import unittest
import time
from web3_bridge import Web3Bridge

class Web3BridgeTestCase(unittest.TestCase):
    def setUp(self):
        self.bridge = Web3Bridge()

    def test_bridge_status(self):
        status = self.bridge.get_status()
        self.assertTrue(status['connected'])
        self.assertIsNotNone(status['contract_address'])
        self.assertIsNotNone(status['terminal_address'])
        self.assertGreaterEqual(status['block_number'], 0)

    def test_mark_attendance_on_chain_flow(self):
        student_id = "STU_WEB3_001"
        now_ts = int(time.time())

        # First dispatch -> should succeed
        res = self.bridge.mark_attendance_on_chain(student_id, now_ts, location_id="ROOM_A")
        self.assertTrue(res['success'])
        self.assertIsNotNone(res['tx_hash'])
        self.assertTrue(res['tx_hash'].startswith("0x"))
        self.assertGreater(res['block_number'], 0)
        self.assertGreater(res['gas_used'], 0)
        self.assertGreater(res['latency_ms'], 0)
        self.assertIsNone(res['error'])

        # Duplicate dispatch -> should fail on-chain
        res_dup = self.bridge.mark_attendance_on_chain(student_id, now_ts, location_id="ROOM_A")
        self.assertFalse(res_dup['success'])
        self.assertIsNotNone(res_dup['error'])
        self.assertTrue("reverted" in res_dup['error'].lower() or "already marked" in res_dup['error'].lower())

if __name__ == '__main__':
    unittest.main()
