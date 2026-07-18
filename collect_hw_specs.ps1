# collect_hw_specs.ps1
# Collects full hardware specs for reproducible ML/CPU-inference papers
# Run as Administrator for complete info (not strictly required)

$outFile = ".\system_specs_$(Get-Date -Format 'yyyyMMdd_HHmmss').txt"

"=============================================" | Out-File $outFile
" HARDWARE SPECIFICATION REPORT"               | Out-File $outFile -Append
" Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $outFile -Append
" Hostname : $env:COMPUTERNAME"                | Out-File $outFile -Append
"=============================================" | Out-File $outFile -Append
"" | Out-File $outFile -Append

# ---------- 1. OPERATING SYSTEM ----------
"[1] OPERATING SYSTEM" | Out-File $outFile -Append
Get-CimInstance Win32_OperatingSystem | Select-Object `
    Caption, Version, BuildNumber, OSArchitecture, `
    @{N='TotalVisibleMemory_GB';E={[math]::Round($_.TotalVisibleMemorySize/1MB,2)}}, `
    @{N='FreePhysicalMemory_GB';E={[math]::Round($_.FreePhysicalMemory/1MB,2)}} `
    | Format-List | Out-String | Out-File $outFile -Append

# ---------- 2. CPU ----------
"[2] CPU (Processor)" | Out-File $outFile -Append
Get-CimInstance Win32_Processor | Select-Object `
    Name, Manufacturer, `
    @{N='Architecture';E={$_.Architecture}}, `
    NumberOfCores, NumberOfLogicalProcessors, `
    @{N='MaxClock_MHz';E={$_.MaxClockSpeed}}, `
    L2CacheSize, L3CacheSize, `
    SocketDesignation, DeviceID, `
    @{N='NUMA_Nodes';E={(Get-CimInstance Win32_Processor).Count}} `
    | Format-List | Out-String | Out-File $outFile -Append

# ---------- 3. CACHE HIERARCHY ----------
"[3] CACHE HIERARCHY (L1/L2/L3)" | Out-File $outFile -Append
Get-CimInstance Win32_CacheMemory | Select-Object `
    Level, Purpose, InstalledSize, MaxCacheSize, Associativity `
    | Format-Table -AutoSize | Out-String | Out-File $outFile -Append

# ---------- 4. MEMORY (RAM) ----------
"[4] PHYSICAL MEMORY (RAM sticks)" | Out-File $outFile -Append
Get-CimInstance Win32_PhysicalMemory | Select-Object `
    BankLabel, DeviceLocator, `
    @{N='Capacity_GB';E={$_.Capacity/1GB}}, `
    @{N='Speed_MHz';E={$_.ConfiguredClockSpeed}}, `
    Manufacturer, PartNumber, `
    @{N='MemoryType';E={switch($_.SMBIOSMemoryType){
        20{'DDR'} 21{'DDR2'} 24{'DDR3'} 26{'DDR4'} 30{'LPDDR4'}
        34{'DDR5'} 35{'LPDDR5'} default{"Type_$($_.SMBIOSMemoryType)"}
    }}}, `
    DataWidth, TotalWidth `
    | Format-Table -AutoSize | Out-String | Out-File $outFile -Append

"[4b] MEMORY CHANNELS SUMMARY" | Out-File $outFile -Append
$sticks = (Get-CimInstance Win32_PhysicalMemory).Count
"Number of populated DIMM slots: $sticks" | Out-File $outFile -Append
"NOTE: To determine actual channel mode (Single/Dual/Quad), use CPU-Z or BIOS." | Out-File $outFile -Append
"" | Out-File $outFile -Append

# ---------- 5. NUMA TOPOLOGY ----------
"[5] NUMA TOPOLOGY" | Out-File $outFile -Append
$cpuCount = (Get-CimInstance Win32_Processor).Count
"Number of physical CPU packages (NUMA nodes in typical config): $cpuCount" | Out-File $outFile -Append
"" | Out-File $outFile -Append

# ---------- 6. CPU FEATURES ----------
"[6] CPU INSTRUCTION SET FLAGS" | Out-File $outFile -Append
# Use CoreInfo-like info via WMI + registry fallback
Get-CimInstance Win32_Processor | Select-Object Name, Description, `
    @{N='VirtualizationFirmwareEnabled';E={$_.VirtualizationFirmwareEnabled}} `
    | Format-List | Out-String | Out-File $outFile -Append
"NOTE: For AVX2/AVX-512/AMX flags, run CoreInfo64.exe -f (Sysinternals) or CPU-Z." | Out-File $outFile -Append
"" | Out-File $outFile -Append

# ---------- 7. SYSTEMINFO (comprehensive snapshot) ----------
"[7] FULL SYSTEMINFO DUMP" | Out-File $outFile -Append
systeminfo | Out-File $outFile -Append

# ---------- 8. POWER / THERMAL PROFILE ----------
"[8] POWER PLAN (affects CPU boost behaviour)" | Out-File $outFile -Append
powercfg /getactivescheme | Out-File $outFile -Append
"" | Out-File $outFile -Append

# ---------- 9. ENVIRONMENT ----------
"[9] RUNTIME ENVIRONMENT" | Out-File $outFile -Append
"PowerShell version: $($PSVersionTable.PSVersion)" | Out-File $outFile -Append
"Python (if installed):" | Out-File $outFile -Append
try { python --version 2>&1 | Out-File $outFile -Append } catch { "  not found" | Out-File $outFile -Append }
"" | Out-File $outFile -Append

"=============================================" | Out-File $outFile -Append
" End of report. File saved to: $outFile" | Out-File $outFile -Append
"=============================================" | Out-File $outFile -Append

Write-Host "✅ Report saved to: $outFile" -ForegroundColor Green
Write-Host "   Attach 'system_specs_*.txt' to your paper's Appendix." -ForegroundColor Cyan