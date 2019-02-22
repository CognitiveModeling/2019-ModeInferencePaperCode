/*******************************************************************************
 * JANNLab Neural Network Framework for Java
 * Copyright (C) 2012-2014 Sebastian Otte
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 ******************************************************************************/

package de.jannlab.tools;


/**
 * This class provides methods for handling strings, string lines
 * and string tables.
 * <br></br>
 * @author Sebastian Otte
 *
 */
public final class StringTools {
    
    public static final char CHAR_SPACE = ' ';
    public static final char CHAR_TAB   = '\t';
    public static final char CHAR_ALIGN = '&';
    
    public static final String SPACE     = String.valueOf(CHAR_SPACE);
    public static final String LINEBREAK  = "\n";
    public static final String LINEBREAK2 = LINEBREAK + LINEBREAK;
    public static final String INDENT    = "    ";
    public static final String ALIGN     =  String.valueOf(CHAR_ALIGN); 
    
    public static final int DEFAULT_TABSPACES      = 4;
    public static final int DEFAULT_SMALLPADDING   = 2;
    public static final int DEFAULT_LARGEPADDING   = 4;
    public static final int DEFAULT_COLUMNWIDTH    = 80;
    public static final int DEFAULT_WORDBREAKWIDTH = 20;
    
    public static final String REGEX_WITHESPACE    = "\\[\t\\x0b]*(,|\\[\t\\x0b])\\[\t\\x0b]*";
    public static final String REGEX_WHITESPACEALL = "\\s*(,|\\s)\\s*";
    
    /**
     * Replaces variables within Strings by given substitutes.
     * @param str Source string containing variables \"#i \" with i from {1, 2, ...}.
     * @param args Substitutes.
     * @return String with substituted variables.
     */
    public static String var(
        final String str, 
        final String ...args
    ) {
        final String key = "#";
        String result = str;
        for (int i = 0; i < args.length; i++) {
            result = result.replace(key + (i + 1), args[i]);
        }
        return result;
    }
    
    /**
     * Merges an array of Strings into one String with
     * some "glue".
     * <br></br>
     * @param s The array of strings.
     * @param glue The glue.
     * @return The merged string.
     */
    public static String combine(final String[] s, final String glue) {
        //
        final int k = s.length;
        //
        if (k == 0) {
            return "";
        }
        //
        final StringBuilder out = new StringBuilder();
        out.append(s[0]);
        //
        for (int x = 1; x < k; x++) {
            out.append(glue).append(s[x]);
        }
        return out.toString();
    }
    
    public static String combine(final String[] s, final String prefix, final String suffix, final String glue) {
        //
        final int k = s.length;
        //
        if (k == 0) {
            return "";
        }
        //
        final StringBuilder out = new StringBuilder();
        if (prefix != null) out.append(prefix);
        out.append(s[0]);
        if (suffix != null) out.append(suffix);
        //
        for (int x = 1; x < k; x++) {
            out.append(glue);
            if (prefix != null) out.append(prefix);
            out.append(s[x]);
            if (suffix != null) out.append(suffix);
            
        }
        return out.toString();
    }

    
    
    /**
     * Indents all lines in a given String.
     * <br></br>
     * @param lines A string containing lines.
     * @return All lines indented.
     */
    public static String indent(final String lines) {
        String[] split = lines.split(LINEBREAK);
        for (int i = 0; i < split.length; i++) {
            split[i] = INDENT + split[i];
        }
        return combine(split, LINEBREAK);
    }
    
    public static String repeat(final String pattern, final int num) {
        StringBuilder out = new StringBuilder();
        for (int i = 0; i < num; i++) {
            out.append(pattern);
        }
        return out.toString();
    }
    
    
    public static String spaces(final int num) {
        return repeat(" ", num);
    }
    
    
    public static String centerColumn(
        final int columnwidth,
        final String column
    ) {
        final String[] lines = column.split(LINEBREAK);
        return centerColumn(columnwidth, lines);    
    }
    
    public static String centerColumn(
        final int columnwidth,
        final String ...lines
    ) {
        final int linesnum      = lines.length;
        final StringBuilder out = new StringBuilder();
        //
        for (int i = 0; i < linesnum; i++) {
            if (i > 0) out.append(LINEBREAK);
            //
            final String line    = lines[i].trim();
            final int linelength = line.length();
            final int padding    = (columnwidth - linelength) / 2;
            //
            out.append(spaces(padding));
            out.append(line);
        }
        //
        return out.toString();

    }
    
    
    public static String alignColumns(
        String ...columns
    ) {
        return alignColumns(DEFAULT_LARGEPADDING, columns);
    }
    
    public static String alignColumns(
        final int padding,
        String ...columns
    ) {
        String[][]  matrix = new String[columns.length][];
        final int[] widths = new int[columns.length];
        int   lines  = 0;
        //
        for (int j = 0; j < columns.length; j++) {
            matrix[j] = columns[j].split(LINEBREAK);
            if (matrix[j].length > lines) {
                lines = matrix[j].length;
            }
            for (int i = 0; i < matrix[j].length; i++) {
                final String s = matrix[j][i];
                if (s.length() > widths[j]) {
                    widths[j] = s.length();
                }
            }
        }
        //
        final StringBuilder out = new StringBuilder();
        //
        for (int i = 0; i < lines; i++) {
            if (i > 0) out.append(LINEBREAK);
            for (int j = 0; j < columns.length; j++) {
                String item = "";
                if (i < matrix[j].length) {
                    item = matrix[j][i];
                }
                //
                out.append(item);
                //
                if (j < (columns.length - 1)) {
                    out.append(spaces(padding + (widths[j] - item.length())));
                }
            }
        }

        return out.toString();
    }
    
    
    public static String leadingWhiteSpaces(final String line) {
        return leadingWhiteSpaces(line, true, DEFAULT_TABSPACES);
    }
    
    public static String leadingWhiteSpaces(final String line, final boolean replacetab, final int tabwidth) {
        //
        int spaces = 0;
        int tabs   = 0;
        int count  = 0;
        //
        for (int i = 0; i < line.length(); i++) {
            final char c = line.charAt(i);
            if (c == CHAR_SPACE) {
                spaces++;
                count++;
            } else if (c == CHAR_TAB) {
                tabs++;
                count++;
            } else {
                break;
            }
        }
        //
        if (replacetab) {
            return spaces(spaces + (tabs * tabwidth));
        } else {
            return line.substring(0, count);
        }
    }
    
    public static String alignColumn(final String text) {
        return alignColumn(text, DEFAULT_COLUMNWIDTH);
    }
    
    public static String alignColumn(final String text, final int columnwidth) {
        //
        final String[]      lines = text.split(LINEBREAK);
        final StringBuilder out   = new StringBuilder();
        //
        for (String line : lines) {
            int    width       = 0;
            String indent      = leadingWhiteSpaces(line);
            int    indentchars = indent.length();
            //
            final String[] token  = line.trim().split(SPACE);
            int consumed          = 0;
            boolean newline       = true;
            //
            while (consumed < token.length) {
                //
                if (token[consumed].startsWith(String.valueOf(CHAR_ALIGN))) {
                    //
                    indent      = spaces(width + 1);
                    indentchars = width + 1;
                    //
                    if (token[consumed].length() == 1) {
                        consumed++;
                    } else {
                        token[consumed] = token[consumed].substring(1);
                    }
                }
                //
                final String t       = token[consumed];
                final int    tlength = t.length(); 
                //
                if (newline) {
                    out.append(indent);
                    width += indentchars;
                } else {
                    if (width < columnwidth) {
                        out.append(CHAR_SPACE);
                        width += 1;
                    }
                }
                //
                if ((width + tlength) <= columnwidth) {
                    out.append(t);
                    width   += tlength;
                    consumed++;
                    newline = false;
                } else {
                    if (tlength > (DEFAULT_WORDBREAKWIDTH)) {
                        final String head = t.substring(0, columnwidth - width);
                        out.append(head);
                        final String rt = t.substring(columnwidth - width, t.length());
                        token[consumed] = rt;
                    }
                    out.append(LINEBREAK);
                    newline = true;
                    width   = 0;
                }
            }
            if (!newline) {
                out.append(LINEBREAK);
            }
        }
        //
        return out.toString();
    }
    
    public static String fitStringLeft(final String str, final int width) {
        final int length = str.length();
        if (length > width) {
            return str.substring(0, width);
        }
        return str + spaces(width - length);
    }
}
