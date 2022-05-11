$testDir=(Get-Item ../../test/simulation/math).FullName
Get-ChildItem $testDir -Filter *.mo |
Foreach-Object {
    $filename = $_.FullName
    ./marco.bat --omc-bypass --end-time=1 -o simulation --model=$filename $filename > $null
    ./simulation.exe | C:\llvm\bin\filecheck.exe $filename
}
del simulation.*