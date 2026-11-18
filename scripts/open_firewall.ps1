# Run ONCE in an *Administrator* PowerShell on Laptop 1 (the API laptop).
# Allows other laptops on the same network to reach the API on TCP port 8000.
New-NetFirewallRule -DisplayName "PathSnap API 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow -Profile Any
# Undo later with:  Remove-NetFirewallRule -DisplayName "PathSnap API 8000"
