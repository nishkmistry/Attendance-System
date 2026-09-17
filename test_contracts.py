import unittest
import time
import solcx
from web3 import Web3
from web3.providers.eth_tester import EthereumTesterProvider
from eth_tester import EthereumTester

class AttendanceContractTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            solcx.install_solc('0.8.20')
        except Exception:
            pass

        with open('contracts/Attendance.sol', 'r') as f:
            contract_source = f.read()

        compiled = solcx.compile_source(
            contract_source,
            output_values=['abi', 'bin'],
            solc_version='0.8.20'
        )
        contract_id, contract_interface = compiled.popitem()

        cls.abi = contract_interface['abi']
        cls.bytecode = contract_interface['bin']

    def setUp(self):
        self.eth_tester = EthereumTester()
        self.w3 = Web3(EthereumTesterProvider(self.eth_tester))

        self.accounts = self.w3.eth.accounts
        self.owner = self.accounts[0]
        self.terminal = self.accounts[1]
        self.unauthorized_user = self.accounts[2]

        # Deploy contract
        AttendanceContract = self.w3.eth.contract(abi=self.abi, bytecode=self.bytecode)
        tx_hash = AttendanceContract.constructor().transact({'from': self.owner})
        tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

        self.contract = self.w3.eth.contract(
            address=tx_receipt.contractAddress,
            abi=self.abi
        )

        # Authorize terminal
        self.contract.functions.authorizeTerminal(self.terminal).transact({'from': self.owner})

    def test_authorization_controls(self):
        self.assertTrue(self.contract.functions.authorizedTerminals(self.owner).call())
        self.assertTrue(self.contract.functions.authorizedTerminals(self.terminal).call())
        self.assertFalse(self.contract.functions.authorizedTerminals(self.unauthorized_user).call())

        # Revoke terminal
        self.contract.functions.revokeTerminal(self.terminal).transact({'from': self.owner})
        self.assertFalse(self.contract.functions.authorizedTerminals(self.terminal).call())

        # Unauthorized user cannot authorize terminals
        with self.assertRaises(Exception):
            self.contract.functions.authorizeTerminal(self.unauthorized_user).transact({'from': self.unauthorized_user})

    def test_mark_attendance_and_duplicate_prevention(self):
        student_id = "REG1001"
        now_ts = int(time.time())
        loc_hash = Web3.keccak(text="Building_A_Room_101")

        # Check not marked
        self.assertFalse(self.contract.functions.isAttendanceMarked(student_id, now_ts).call())

        # Mark attendance
        tx_hash = self.contract.functions.markAttendance(
            student_id, now_ts, loc_hash
        ).transact({'from': self.terminal})

        tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        self.assertEqual(tx_receipt.status, 1)

        # Confirm marked
        self.assertTrue(self.contract.functions.isAttendanceMarked(student_id, now_ts).call())

        # Check event logs
        events = self.contract.events.AttendanceMarked().get_logs()
        self.assertEqual(len(events), 1)
        expected_indexed_id = Web3.keccak(text=student_id)
        self.assertEqual(events[0]['args']['studentId'], expected_indexed_id)
        self.assertEqual(events[0]['args']['locationHash'], loc_hash)

        # Attempt duplicate mark on same date -> should revert
        with self.assertRaises(Exception):
            self.contract.functions.markAttendance(
                student_id, now_ts, loc_hash
            ).transact({'from': self.terminal})

    def test_unauthorized_address_rejection(self):
        student_id = "REG1002"
        now_ts = int(time.time())
        loc_hash = Web3.keccak(text="Building_A_Room_101")

        with self.assertRaises(Exception):
            self.contract.functions.markAttendance(
                student_id, now_ts, loc_hash
            ).transact({'from': self.unauthorized_user})

    def test_gas_usage_individual_vs_batched(self):
        student_ids = ["REG2001", "REG2002", "REG2003"]
        now_ts = int(time.time())
        loc_hash = Web3.keccak(text="Building_B_Room_202")

        # Individual transactions
        gas_individual_total = 0
        for sid in student_ids:
            tx_hash = self.contract.functions.markAttendance(
                sid, now_ts, loc_hash
            ).transact({'from': self.terminal})
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            gas_individual_total += receipt.gasUsed

        # Batch transaction for another set of students
        batch_ids = ["REG3001", "REG3002", "REG3003"]
        timestamps = [now_ts] * len(batch_ids)
        hashes = [loc_hash] * len(batch_ids)

        tx_hash_batch = self.contract.functions.batchMarkAttendance(
            batch_ids, timestamps, hashes
        ).transact({'from': self.terminal})
        receipt_batch = self.w3.eth.wait_for_transaction_receipt(tx_hash_batch)
        gas_batched_total = receipt_batch.gasUsed

        # Batching should be more gas efficient per student
        self.assertLess(gas_batched_total, gas_individual_total)
        print(f"\n[Gas Analysis] Individual Total for {len(student_ids)}: {gas_individual_total} gas ({gas_individual_total/len(student_ids):.0f}/call)")
        print(f"[Gas Analysis] Batched Total for {len(batch_ids)}: {gas_batched_total} gas ({gas_batched_total/len(batch_ids):.0f}/call)")

if __name__ == '__main__':
    unittest.main()
