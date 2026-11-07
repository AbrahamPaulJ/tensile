// A private chain inside the page: the EthereumJS VM at the Osaka hardfork, the hardfork the gas
// figures in docs/ were measured on with anvil. Transactions are real signed transactions through
// runTx, so gasUsed includes the base cost, the calldata and the EIP-7623 floor. The gas price is
// zero, so balances change only by what the contracts move.
import { createBlock } from '@ethereumjs/block';
import { Common, Hardfork, Mainnet } from '@ethereumjs/common';
import { createLegacyTx } from '@ethereumjs/tx';
import {
  Account,
  Address,
  bytesToHex,
  createAddressFromPrivateKey,
  createContractAddress,
  createAddressFromString,
  hexToBytes,
} from '@ethereumjs/util';
import { createVM, runTx } from '@ethereumjs/vm';

const TX_GAS_CAP = 16_777_216n; // EIP-7825
const BLOCK_GAS = 60_000_000n;
const GENESIS_TIME = 1_800_000_000n;

export class Chain {
  static async create(names) {
    const chain = new Chain();
    chain.common = new Common({ chain: Mainnet, hardfork: Hardfork.Osaka });
    chain.vm = await createVM({ common: chain.common });
    chain.now = GENESIS_TIME;
    chain.number = 1n;
    chain.accounts = {};
    let index = 1;
    for (const name of names) {
      // Well-known throwaway keys: the first is anvil's default account, as in tools/arbiter_gas.sh.
      const key = hexToBytes(name.key ?? `0x${index.toString(16).padStart(64, '0')}`);
      const address = createAddressFromPrivateKey(key);
      await chain.vm.stateManager.putAccount(
        address,
        new Account(0n, name.balance ?? 100n * 10n ** 18n),
      );
      chain.accounts[name.name] = { key, address, nonce: 0n, label: name.name };
      index++;
    }
    return chain;
  }

  block() {
    return createBlock(
      {
        header: {
          number: this.number,
          timestamp: this.now,
          gasLimit: BLOCK_GAS,
          baseFeePerGas: 0n,
          excessBlobGas: 0n,
          blobGasUsed: 0n,
          parentBeaconBlockRoot: new Uint8Array(32),
          requestsHash: new Uint8Array(32),
        },
      },
      { common: this.common, freeze: false },
    );
  }

  /** A signed transaction. Returns what a receipt and a debug trace together would. */
  async send({ from, to = null, data, value = 0n }) {
    const account = this.accounts[from];
    const tx = createLegacyTx(
      {
        nonce: account.nonce,
        gasPrice: 0n,
        gasLimit: TX_GAS_CAP,
        to: to ? to : undefined,
        value,
        data,
      },
      { common: this.common },
    ).sign(account.key);
    const created = to ? null : createContractAddress(account.address, account.nonce);
    const started = performance.now();
    const result = await runTx(this.vm, { tx, block: this.block(), skipHardForkValidation: true });
    const ms = performance.now() - started;
    account.nonce += 1n;
    this.number += 1n;
    this.now += 12n;
    const error = result.execResult.exceptionError;
    return {
      gasUsed: result.totalGasSpent,
      reverted: Boolean(error),
      error: error ? String(error.error ?? error) : null,
      returnValue: result.execResult.returnValue,
      logs: result.execResult.logs ?? [],
      address: created,
      ms,
    };
  }

  /** A call that changes nothing, as eth_call. */
  async call({ to, data, from = null }) {
    const caller = from ? this.accounts[from].address : Object.values(this.accounts)[0].address;
    await this.vm.stateManager.checkpoint();
    const started = performance.now();
    try {
      const result = await this.vm.evm.runCall({
        to,
        caller,
        origin: caller,
        data,
        gasLimit: TX_GAS_CAP,
        block: this.block(),
      });
      const error = result.execResult.exceptionError;
      return {
        reverted: Boolean(error),
        returnValue: result.execResult.returnValue,
        executionGas: result.execResult.executionGasUsed,
        ms: performance.now() - started,
      };
    } finally {
      await this.vm.stateManager.revert();
    }
  }

  async balance(address) {
    const account = await this.vm.stateManager.getAccount(address);
    return account ? account.balance : 0n;
  }

  /** Moves the clock, as anvil's evm_increaseTime. */
  advance(seconds) {
    this.now += BigInt(seconds);
  }
}

export { Address, bytesToHex, createAddressFromString };
