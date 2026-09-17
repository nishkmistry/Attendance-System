// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title Attendance Smart Contract
 * @notice Manages student attendance records on-chain with authorized terminal access control.
 */
contract Attendance {
    address public owner;
    mapping(address => bool) public authorizedTerminals;

    // Student ID => Date Timestamp (day boundary epoch seconds) => Status
    mapping(string => mapping(uint256 => bool)) public hasAttended;

    event AttendanceMarked(string indexed studentId, uint256 timestamp, bytes32 locationHash);
    event TerminalAuthorized(address indexed terminal);
    event TerminalRevoked(address indexed terminal);

    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can perform this action");
        _;
    }

    modifier onlyAuthorizedTerminal() {
        require(authorizedTerminals[msg.sender] || msg.sender == owner, "Not an authorized terminal");
        _;
    }

    constructor() {
        owner = msg.sender;
        authorizedTerminals[msg.sender] = true;
        emit TerminalAuthorized(msg.sender);
    }

    function authorizeTerminal(address terminal) external onlyOwner {
        require(terminal != address(0), "Invalid terminal address");
        authorizedTerminals[terminal] = true;
        emit TerminalAuthorized(terminal);
    }

    function revokeTerminal(address terminal) external onlyOwner {
        authorizedTerminals[terminal] = false;
        emit TerminalRevoked(terminal);
    }

    function getDayTimestamp(uint256 timestamp) public pure returns (uint256) {
        return timestamp - (timestamp % 86400);
    }

    function markAttendance(
        string memory studentId,
        uint256 timestamp,
        bytes32 locationHash
    ) public onlyAuthorizedTerminal {
        uint256 dayTimestamp = getDayTimestamp(timestamp);
        require(!hasAttended[studentId][dayTimestamp], "Attendance already marked for this date");

        hasAttended[studentId][dayTimestamp] = true;
        emit AttendanceMarked(studentId, timestamp, locationHash);
    }

    function batchMarkAttendance(
        string[] memory studentIds,
        uint256[] memory timestamps,
        bytes32[] memory locationHashes
    ) external onlyAuthorizedTerminal {
        require(
            studentIds.length == timestamps.length && timestamps.length == locationHashes.length,
            "Mismatched input lengths"
        );

        for (uint256 i = 0; i < studentIds.length; i++) {
            markAttendance(studentIds[i], timestamps[i], locationHashes[i]);
        }
    }

    function isAttendanceMarked(string memory studentId, uint256 timestamp) external view returns (bool) {
        uint256 dayTimestamp = getDayTimestamp(timestamp);
        return hasAttended[studentId][dayTimestamp];
    }
}
