import time
import os
import json
import solcx
from web3 import Web3
from eth_account import Account
from web3.providers.eth_tester import EthereumTesterProvider
from eth_tester import EthereumTester

class Web3Bridge:
    def __init__(self, rpc_url=None, private_key=None, contract_address=None):
        self.rpc_url = rpc_url or os.environ.get('RPC_URL')
        self.private_key = private_key or os.environ.get('TERMINAL_PRIVATE_KEY')
        self.contract_address = contract_address or os.environ.get('CONTRACT_ADDRESS')

        self._init_web3()
        self._init_account()
        self._load_or_deploy_contract()

    def _init_web3(self):
        if self.rpc_url and self.rpc_url.startswith('http'):
            self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        else:
            self.eth_tester = EthereumTester()
            self.w3 = Web3(EthereumTesterProvider(self.eth_tester))

    def _init_account(self):
        if self.private_key:
            self.account = Account.from_key(self.private_key)
        else:
            test_acc = self.w3.eth.accounts[0] if self.w3.eth.accounts else Account.create().address
            self.account = Account.from_key('0x4f3edf983ac636a65a842ce7c78d9aa706d3b113bce9c46f30d7d21715b23b1d')
            if hasattr(self, 'eth_tester'):
                self.w3.eth.send_transaction({
                    'from': test_acc,
                    'to': self.account.address,
                    'value': self.w3.to_wei(10, 'ether')
                })

    def _compile_contract(self):
        contract_path = os.path.join(os.path.dirname(__file__), 'contracts', 'Attendance.sol')
        try:
            solcx.install_solc('0.8.20')
        except Exception:
            pass

        with open(contract_path, 'r') as f:
            source = f.read()

        compiled = solcx.compile_source(
            source,
            output_values=['abi', 'bin'],
            solc_version='0.8.20'
        )
        _, contract_interface = compiled.popitem()
        return contract_interface['abi'], contract_interface['bin']

    def _load_or_deploy_contract(self):
        self.abi, self.bytecode = self._compile_contract()

        if self.contract_address:
            self.contract = self.w3.eth.contract(address=self.contract_address, abi=self.abi)
        else:
            Attendance = self.w3.eth.contract(abi=self.abi, bytecode=self.bytecode)
            nonce = self.w3.eth.get_transaction_count(self.account.address)

            construct_tx = Attendance.constructor().build_transaction({
                'from': self.account.address,
                'nonce': nonce,
                'gas': 3000000,
                'gasPrice': self.w3.eth.gas_price
            })

            signed = self.account.sign_transaction(construct_tx)
            raw_tx = getattr(signed, 'raw_transaction', getattr(signed, 'rawTransaction', None))
            tx_hash = self.w3.eth.send_raw_transaction(raw_tx)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

            self.contract_address = receipt.contractAddress
            self.contract = self.w3.eth.contract(address=self.contract_address, abi=self.abi)

    def mark_attendance_on_chain(self, student_id, epoch_timestamp=None, location_id="MAIN_TERMINAL"):
        """
        Constructs, signs, and broadcasts markAttendance transaction.
        Returns result dict containing tx_hash, block_number, gas_used, and latency.
        """
        if epoch_timestamp is None:
            epoch_timestamp = int(time.time())

        location_hash = Web3.keccak(text=location_id)

        start_time = time.time()
        try:
            nonce = self.w3.eth.get_transaction_count(self.account.address)
            tx = self.contract.functions.markAttendance(
                student_id, epoch_timestamp, location_hash
            ).build_transaction({
                'from': self.account.address,
                'nonce': nonce,
                'gas': 200000,
                'gasPrice': self.w3.eth.gas_price
            })

            signed_tx = self.account.sign_transaction(tx)
            raw_tx = getattr(signed_tx, 'raw_transaction', getattr(signed_tx, 'rawTransaction', None))

            tx_hash = self.w3.eth.send_raw_transaction(raw_tx)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            latency_ms = round((time.time() - start_time) * 1000, 2)

            tx_hash_hex = Web3.to_hex(receipt.transactionHash)
            is_success = (receipt.status == 1)
            err_msg = None if is_success else "Transaction reverted on-chain (e.g. duplicate attendance or unauthorized terminal)"

            return {
                'success': is_success,
                'tx_hash': tx_hash_hex,
                'block_number': receipt.blockNumber,
                'gas_used': receipt.gasUsed,
                'latency_ms': latency_ms,
                'location_hash': Web3.to_hex(location_hash),
                'student_id': student_id,
                'timestamp': epoch_timestamp,
                'error': err_msg
            }
        except Exception as e:
            latency_ms = round((time.time() - start_time) * 1000, 2)
            return {
                'success': False,
                'tx_hash': None,
                'block_number': None,
                'gas_used': 0,
                'latency_ms': latency_ms,
                'location_hash': Web3.to_hex(location_hash),
                'student_id': student_id,
                'timestamp': epoch_timestamp,
                'error': str(e)
            }

    def get_status(self):
        """Returns Web3 connection and terminal details."""
        return {
            'connected': self.w3.is_connected() if hasattr(self.w3, 'is_connected') else True,
            'contract_address': self.contract_address,
            'terminal_address': self.account.address,
            'block_number': self.w3.eth.block_number,
            'network_id': getattr(self.w3.eth, 'chain_id', 1337)
        }
