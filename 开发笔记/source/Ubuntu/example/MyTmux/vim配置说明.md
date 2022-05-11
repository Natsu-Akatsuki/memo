# vim配置说明

## [which-key](https://github.com/liuchengxu/vim-which-key)

```bash
# 安装which-key插件
call plug#begin('~/.vim/plugged')
Plug 'liuchengxu/vim-which-key'
call plug#end()

call which_key#register('<Space>', "g:which_key_map")
" 映射WhichKey按键（设置前导符）
let g:mapleader = "\<Space>"
let g:maplocalleader = ','

nnoremap <silent> <leader> :<c-u>WhichKey '<Space>'<CR>
vnoremap <silent> <leader> :<c-u>WhichKeyVisual '<Space>'<CR>

" 设置响应时间
set timeoutlen=250

" Define prefix dictionary
let g:which_key_map =  {}

" Second level dictionaries:
" 'name' is a special field. It will define the name of the group, e.g., leader-f is the "+file" group.
" Unnamed groups will show a default empty string.

" =======================================================
" Create menus based on existing mappings
" =======================================================
" You can pass a descriptive text to an existing mapping.

let g:which_key_map.f = { 'name' : '+file' }

nnoremap <silent> <leader>fs :update<CR>
let g:which_key_map.f.s = 'save-file'

nnoremap <silent> <leader>fd :e $MYVIMRC<CR>
let g:which_key_map.f.d = 'open-vimrc'

nnoremap <silent> <leader>oq  :copen<CR>
nnoremap <silent> <leader>ol  :lopen<CR>
let g:which_key_map.o = { 
      \ 'name' : '+open',
      \ 'q' : 'open-quickfix'    ,   
      \ 'l' : 'open-locationlist',
      \ }

" =======================================================
" Create menus not based on existing mappings:
" =======================================================
" Provide commands(ex-command, <Plug>/<C-W>/<C-d> mapping, etc.)
" and descriptions for the existing mappings.
"
" Note:
" Some complicated ex-cmd may not work as expected since they'll be
" feed into `feedkeys()`, in which case you have to define a decicated
" Command or function wrapper to make it work with vim-which-key.
" Ref issue #126, #133 etc.
let g:which_key_map.b = { 
      \ 'name' : '+buffer' ,
      \ '1' : ['b1'        , 'buffer 1']        ,   
      \ '2' : ['b2'        , 'buffer 2']        ,   
      \ 'd' : ['bd'        , 'delete-buffer']   ,   
      \ 'f' : ['bfirst'    , 'first-buffer']    ,   
      \ 'h' : ['Startify'  , 'home-buffer']     ,   
      \ 'l' : ['blast'     , 'last-buffer']     ,   
      \ 'n' : ['bnext'     , 'next-buffer']     ,   
      \ 'p' : ['bprevious' , 'previous-buffer'] ,
      \ '?' : ['Buffers'   , 'fzf-buffer']      ,   
      \ }

let g:which_key_map.l = { 
      \ 'name' : '+lsp',
      \ 'f' : ['spacevim#lang#util#Format()'          , 'formatting']       ,   
      \ 'r' : ['spacevim#lang#util#FindReferences()'  , 'references']       ,   
      \ 'R' : ['spacevim#lang#util#Rename()'          , 'rename']           ,   
      \ 's' : ['spacevim#lang#util#DocumentSymbol()'  , 'document-symbol']  ,
      \ 'S' : ['spacevim#lang#util#WorkspaceSymbol()' , 'workspace-symbol'] ,
      \ 'g' : { 
        \ 'name': '+goto',
        \ 'd' : ['spacevim#lang#util#Definition()'     , 'definition']      ,   
        \ 't' : ['spacevim#lang#util#TypeDefinition()' , 'type-definition'] ,
        \ 'i' : ['spacevim#lang#util#Implementation()' , 'implementation']  ,
        \ },
      \ }

let g:which_key_map['w'] = { 
      \ 'name' : '+windows' ,
      \ 'w' : ['<C-W>w'     , 'other-window']          ,   
      \ 'd' : ['<C-W>c'     , 'delete-window']         ,   
      \ '-' : ['<C-W>s'     , 'split-window-below']    ,   
      \ '|' : ['<C-W>v'     , 'split-window-right']    ,   
      \ '2' : ['<C-W>v'     , 'layout-double-columns'] ,
      \ 'h' : ['<C-W>h'     , 'window-left']           ,   
      \ 'j' : ['<C-W>j'     , 'window-below']          ,   
      \ 'l' : ['<C-W>l'     , 'window-right']          ,   
      \ 'k' : ['<C-W>k'     , 'window-up']             ,   
      \ 'H' : ['<C-W>5<'    , 'expand-window-left']    ,   
      \ 'J' : [':resize +5'  , 'expand-window-below']   ,   
      \ 'L' : ['<C-W>5>'    , 'expand-window-right']   ,   
      \ 'K' : [':resize -5'  , 'expand-window-up']      ,   
      \ '=' : ['<C-W>='     , 'balance-window']        ,   
      \ 's' : ['<C-W>s'     , 'split-window-below']    ,   
      \ 'v' : ['<C-W>v'     , 'split-window-below']    ,   
      \ '?
```

## [space-vim-dark(theme)](https://github.com/liuchengxu/space-vim-dark)

```bash
call plug#begin('~/.vim/plugged')
Plug 'liuchengxu/space-vim-dark'
call plug#end()

colorscheme space-vim-dark
# 设置comment的颜色为灰色调
hi Comment guifg=#5C6370 ctermfg=59
hi LineNr ctermbg=NONE guibg=NONE
```

## [vim-airline](https://github.com/vim-airline/vim-airline)

- [可选主题](https://github.com/vim-airline/vim-airline/wiki/Screenshots)

```bash
call plug#begin('~/.vim/plugged')
Plug 'vim-airline/vim-airline'
Plug 'vim-airline/vim-airline-themes'
call plug#end()

let g:airline_theme='badwolf'
```

s

```
function! AirlineInit()
	let g:airline_section_a = airline#section#create(['mode',' ','branch'])
	let g:airline_section_b = airline#section#create_left(['ffenc','hunks','%f'])
	let g:airline_section_c = airline#section#create(['filetype'])
	let g:airline_section_x = airline#section#create(['%P'])
	let g:airline_section_y = airline#section#create(['%B'])
	let g:airline_section_z = airline#section#create_right(['%l','%c'])
endfunction
autocmd VimEnter * call AirlineInit()
```

