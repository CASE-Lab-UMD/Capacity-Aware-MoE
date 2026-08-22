import re

with open('docs/index.html', 'r', encoding='utf-8') as f:
    html = f.read()

# 1. Update #simulator section wrapper
html = html.replace('<section id="simulator" class="py-16 md:py-24 bg-slate-900 text-white relative overflow-hidden">',
                    '<section id="simulator" class="py-16 md:py-24 bg-slate-50 text-slate-900 border-b border-slate-200 relative overflow-hidden">')

html = html.replace('<div class="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-500/20 text-emerald-300 text-xs font-semibold uppercase tracking-wider mb-3">',
                    '<div class="inline-flex items-center gap-1.5 px-3.5 py-1 rounded-full bg-emerald-50 border border-emerald-200 text-emerald-800 text-xs font-bold uppercase tracking-wider mb-3">')

html = html.replace('<h2 class="font-heading font-bold text-3xl sm:text-4xl tracking-tight text-white mb-4">',
                    '<h2 class="font-heading font-extrabold text-3xl sm:text-4xl tracking-tight text-slate-900 mb-4">')

html = html.replace('<p class="text-slate-300 text-sm sm:text-base">',
                    '<p class="text-slate-600 text-base leading-relaxed">')

# 2. Simulator App Container
html = html.replace('<div class="bg-slate-800/90 border border-slate-700/80 rounded-3xl p-6 sm:p-8 shadow-2xl backdrop-blur-xl">',
                    '<div class="bg-white border border-slate-200 rounded-3xl p-6 sm:p-8 shadow-sm">')

html = html.replace('border-b border-slate-700/80 pb-8', 'border-b border-slate-200 pb-8')

# Labels & inputs in simulator
html = html.replace('<label class="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">',
                    '<label class="block text-xs font-bold text-slate-700 uppercase tracking-wider mb-2">')

html = html.replace('<label class="text-xs font-semibold text-slate-400 uppercase tracking-wider">',
                    '<label class="text-xs font-bold text-slate-700 uppercase tracking-wider">')

html = html.replace('class="w-full bg-slate-900 border border-slate-600 rounded-xl px-3.5 py-2.5 text-sm text-white font-medium focus:ring-2 focus:ring-emerald-500 focus:border-transparent outline-none transition-all"',
                    'class="w-full bg-white border border-slate-300 rounded-xl px-3.5 py-2.5 text-sm text-slate-800 font-semibold focus:ring-2 focus:ring-emerald-500 focus:border-transparent outline-none transition-all shadow-sm"')

html = html.replace('<p class="text-[11px] text-slate-400 mt-1.5" id="simModelDesc">',
                    '<p class="text-xs text-slate-500 mt-1.5 font-medium" id="simModelDesc">')

html = html.replace('bg-emerald-950/80 px-2 py-0.5 rounded border border-emerald-800',
                    'bg-emerald-50 text-emerald-800 px-2.5 py-0.5 rounded border border-emerald-300')

html = html.replace('bg-indigo-950/80 px-2 py-0.5 rounded border border-indigo-800',
                    'bg-indigo-50 text-indigo-800 px-2.5 py-0.5 rounded border border-indigo-300')

html = html.replace('bg-slate-700 h-2 rounded-lg cursor-pointer',
                    'bg-slate-200 h-2 rounded-lg cursor-pointer')

# Strategy button container
html = html.replace('<div class="grid grid-cols-3 gap-1.5 bg-slate-900 p-1 rounded-xl border border-slate-700">',
                    '<div class="grid grid-cols-3 gap-1.5 bg-slate-100 p-1 rounded-xl border border-slate-200">')

html = html.replace('class="sim-strat-btn text-xs font-semibold py-1.5 rounded-lg transition-all text-slate-400 hover:text-white"',
                    'class="sim-strat-btn text-xs font-semibold py-1.5 rounded-lg transition-all text-slate-700 hover:bg-slate-200/60"')

html = html.replace('<button id="simShuffleBtn" type="button" class="w-full text-xs font-semibold py-1.5 px-3 rounded-lg bg-slate-700 hover:bg-slate-600 text-slate-200 transition-colors flex items-center justify-center gap-1.5">',
                    '<button id="simShuffleBtn" type="button" class="w-full text-xs font-semibold py-1.5 px-3 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-700 border border-slate-300 transition-colors flex items-center justify-center gap-1.5">')

# HUD Stats
html = html.replace('<div class="bg-slate-900/90 border border-slate-700/60 p-4 rounded-2xl">',
                    '<div class="bg-slate-50 border border-slate-200/80 p-4 rounded-2xl">')

html = html.replace('<div class="text-xs text-slate-400 font-medium">',
                    '<div class="text-xs text-slate-600 font-bold uppercase tracking-wider">')

html = html.replace('class="text-xl sm:text-2xl font-mono font-bold text-white mt-0.5"',
                    'class="text-xl sm:text-2xl font-mono font-black text-slate-900 mt-0.5"')

html = html.replace('class="text-xl sm:text-2xl font-mono font-bold text-indigo-400 mt-0.5"',
                    'class="text-xl sm:text-2xl font-mono font-black text-indigo-700 mt-0.5"')

html = html.replace('class="text-xl sm:text-2xl font-mono font-bold text-rose-400 mt-0.5"',
                    'class="text-xl sm:text-2xl font-mono font-black text-rose-700 mt-0.5"')

html = html.replace('class="text-xl sm:text-2xl font-mono font-bold text-emerald-400 mt-0.5"',
                    'class="text-xl sm:text-2xl font-mono font-black text-emerald-700 mt-0.5"')

html = html.replace('<div class="text-[11px] text-slate-400 mt-1">Theoretical ideal balance</div>',
                    '<div class="text-xs text-slate-500 mt-1 font-medium">Theoretical ideal balance</div>')

html = html.replace('<div class="text-[11px] text-indigo-300/80 mt-1">$C = \\lceil \\gamma \\cdot \\bar{N} \\rceil$</div>',
                    '<div class="text-xs text-indigo-700 mt-1 font-semibold">$C = \\lceil \\gamma \\cdot \\bar{N} \\rceil$</div>')

html = html.replace('class="text-[11px] text-rose-300 mt-1"',
                    'class="text-xs text-rose-700 mt-1 font-semibold"')

html = html.replace('class="text-[11px] text-emerald-300 mt-1"',
                    'class="text-xs text-emerald-700 mt-1 font-semibold"')

# Real-time Histogram Card
html = html.replace('<div class="relative bg-slate-950/80 border border-slate-800 rounded-2xl p-6">',
                    '<div class="relative bg-slate-50 border border-slate-200 rounded-2xl p-6">')

html = html.replace('<h3 class="text-sm font-semibold text-white">Distributed Expert Load Profile</h3>',
                    '<h3 class="text-sm font-bold text-slate-900">Distributed Expert Load Profile</h3>')

html = html.replace('text-emerald-400', 'text-emerald-700 font-semibold')
html = html.replace('text-amber-400', 'text-amber-700 font-semibold')
html = html.replace('text-rose-400', 'text-rose-700 font-semibold')
html = html.replace('text-indigo-400', 'text-indigo-700 font-semibold')

html = html.replace('border-b border-slate-800', 'border-b border-slate-300')
html = html.replace('border-indigo-400 pointer-events-none', 'border-indigo-600 pointer-events-none')
html = html.replace('bg-indigo-950 text-indigo-300 text-[10px] font-mono px-1.5 py-0.5 rounded border border-indigo-700',
                    'bg-indigo-50 text-indigo-800 text-xs font-mono font-bold px-2 py-0.5 rounded border border-indigo-300')

# Strategy explainer footer
html = html.replace('<div class="mt-4 p-4 rounded-xl bg-slate-900 border border-slate-800 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 text-xs">',
                    '<div class="mt-4 p-4 rounded-xl bg-white border border-slate-200 shadow-sm flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 text-xs sm:text-sm">')

html = html.replace('<div class="flex items-center gap-2 text-slate-300">',
                    '<div class="flex items-center gap-2 text-slate-700 font-medium">')

html = html.replace('<i class="fa-solid fa-circle-info text-emerald-400 text-sm"></i>',
                    '<i class="fa-solid fa-circle-info text-emerald-600 text-base"></i>')

html = html.replace('class="font-mono text-emerald-400 shrink-0"',
                    'class="font-mono text-emerald-700 font-bold shrink-0"')

# Hero and metric stat updates
html = html.replace('<div class="text-[11px] text-slate-400 mt-0.5">Mixtral-8x7B Speedup</div>',
                    '<div class="text-xs text-slate-500 mt-0.5 font-medium">Mixtral-8x7B Speedup</div>')
html = html.replace('<div class="text-[11px] text-slate-400 mt-0.5">OLMoE-1B-7B Speedup</div>',
                    '<div class="text-xs text-slate-500 mt-0.5 font-medium">OLMoE-1B-7B Speedup</div>')
html = html.replace('<div class="text-[11px] text-slate-400 mt-0.5">Lossless Quality Retention</div>',
                    '<div class="text-xs text-slate-500 mt-0.5 font-medium">Lossless Quality Retention</div>')
html = html.replace('<div class="text-[11px] text-slate-400 mt-0.5">Zero-shot Inference Patch</div>',
                    '<div class="text-xs text-slate-500 mt-0.5 font-medium">Zero-shot Inference Patch</div>')

# Replace all remaining tiny font occurrences
html = html.replace('text-[8px]', 'text-xs')
html = html.replace('text-[9px]', 'text-xs')
html = html.replace('text-[10px]', 'text-xs')
html = html.replace('text-[11px]', 'text-xs')
html = html.replace('text-slate-400', 'text-slate-600')

# JS button classes in simulator
html = html.replace('b.classList.add("text-slate-400");', 'b.classList.add("text-slate-700");')
html = html.replace('btn.classList.remove("text-slate-400");', 'btn.classList.remove("text-slate-700");')

with open('docs/index.html', 'w', encoding='utf-8') as f:
    f.write(html)

with open('index.html', 'w', encoding='utf-8') as f:
    f.write(html)

print('Capacity-Aware-MoE refactored successfully!')
