-- Pandoc filter for the Word export: every underlined span (used only for [todo]
-- placeholders and author review notes) becomes yellow highlighted text.
local function run(text)
  local escaped = text:gsub("&", "&amp;"):gsub("<", "&lt;"):gsub(">", "&gt;")
  return pandoc.RawInline("openxml",
    '<w:r><w:rPr><w:highlight w:val="yellow"/></w:rPr><w:t xml:space="preserve">' .. escaped .. '</w:t></w:r>')
end

function Underline(element)
  local out = {}
  for _, inline in ipairs(element.content) do
    if inline.t == "Str" then
      table.insert(out, run(inline.text))
    elseif inline.t == "Space" or inline.t == "SoftBreak" then
      table.insert(out, run(" "))
    else
      table.insert(out, run(pandoc.utils.stringify(inline)))
    end
  end
  return out
end
